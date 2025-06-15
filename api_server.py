#!/usr/bin/env python3
"""
Simple API server for the LLM Text Reviewer browser extension.
Provides an endpoint to get feedback on text using the existing reviewer functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import tempfile
import yaml
import logging
import re
import google.generativeai as genai
import asyncio
import concurrent.futures
import time

# Add the current directory to Python path to import reviewer
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key=None, fast_model="gemini-2.5-flash-preview-05-20", pro_model="gemini-2.5-pro-preview-06-05"):
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.fast_model = genai.GenerativeModel(fast_model)
        self.pro_model = genai.GenerativeModel(pro_model)
        self.fast_model_name = fast_model
        self.pro_model_name = pro_model
    
    def review_text_content(self, system_prompt, user_prompt, use_pro_model=False):
        """Get feedback from Gemini API."""
        try:
            # Choose model based on importance
            model = self.pro_model if use_pro_model else self.fast_model
            model_name = self.pro_model_name if use_pro_model else self.fast_model_name
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = model.generate_content(full_prompt)
            
            return {
                'response': response.text,
                'model': model_name,
                'error': None
            }
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                'response': '',
                'model': self.pro_model_name if use_pro_model else self.fast_model_name,
                'error': str(e)
            }

    def enhance_suggestion(self, original_text, improved_text, context=""):
        """Use Pro model to enhance high-priority suggestions with better rewrites."""
        try:
            enhance_prompt = f"""You are an expert writing coach. A text improvement suggestion has been identified as high-priority. Please provide an enhanced rewrite that goes beyond the initial suggestion.

Original text: "{original_text}"
Current suggestion: "{improved_text}"
Context: {context if context else "General text improvement"}

Provide a superior rewrite that:
1. Incorporates the spirit of the original suggestion
2. Makes additional improvements for clarity, style, and impact
3. Maintains the original meaning and tone
4. Is significantly better than both the original and current suggestion

Enhanced rewrite:"""

            response = self.pro_model.generate_content(enhance_prompt)
            
            return {
                'enhanced_text': response.text.strip(),
                'model': self.pro_model_name,
                'error': None
            }
        except Exception as e:
            logger.error(f"Gemini Pro enhancement error: {str(e)}")
            return {
                'enhanced_text': improved_text,  # Fallback to original suggestion
                'model': self.pro_model_name,
                'error': str(e)
            }

class SimpleReviewFormatter:
    def __init__(self):
        # Pattern to match suggestions with ratings
        self.suggestion_patterns = [
            # (change "original" -> "improved" [rating: X])
            re.compile(r'\(change\s+"([^"]+)"\s*->\s*"([^"]+)"\s*\[rating:\s*(\d+)\]\)', re.IGNORECASE | re.DOTALL),
            # Change "original" to "improved" [rating: X]
            re.compile(r'change\s+"([^"]+)"\s*to\s*"([^"]+)"\s*\[rating:\s*(\d+)\]', re.IGNORECASE | re.DOTALL),
            # "original" -> "improved" [rating: X]
            re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[rating:\s*(\d+)\]', re.IGNORECASE | re.DOTALL),
            # "original" → "improved" [rating: X] (unicode arrow)
            re.compile(r'"([^"]+)"\s*→\s*"([^"]+)"\s*\[rating:\s*(\d+)\]', re.IGNORECASE | re.DOTALL),
            # Replace "original" with "improved" [rating: X]
            re.compile(r'replace\s+"([^"]+)"\s*with\s*"([^"]+)"\s*\[rating:\s*(\d+)\]', re.IGNORECASE | re.DOTALL)
        ]
        
        # Fallback patterns without ratings (for backward compatibility)
        self.fallback_patterns = [
            re.compile(r'\(change\s+"([^"]+)"\s*->\s*"([^"]+)"\)', re.IGNORECASE | re.DOTALL),
            re.compile(r'change\s+"([^"]+)"\s*to\s*"([^"]+)"', re.IGNORECASE | re.DOTALL),
            re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"', re.IGNORECASE | re.DOTALL),
            re.compile(r'"([^"]+)"\s*→\s*"([^"]+)"', re.IGNORECASE | re.DOTALL),
            re.compile(r'replace\s+"([^"]+)"\s*with\s*"([^"]+)"', re.IGNORECASE | re.DOTALL)
        ]

    def parse_suggestions(self, response):
        """Parse suggestions with ratings from Gemini response."""
        if not response:
            return []
        
        suggestions = []
        
        # First try to find suggestions with ratings
        for pattern in self.suggestion_patterns:
            matches = pattern.findall(response)
            for match in matches:
                if len(match) == 3:  # original, improved, rating
                    original, improved, rating = match
                    original_clean = original.strip()
                    improved_clean = improved.strip()
                    
                    try:
                        rating_int = int(rating)
                        # Clamp rating between 1 and 10
                        rating_int = max(1, min(10, rating_int))
                    except ValueError:
                        rating_int = 5  # Default rating if parsing fails
                    
                    if original_clean and improved_clean and original_clean != improved_clean:
                        suggestions.append({
                            'original': original_clean,
                            'improved': improved_clean,
                            'rating': rating_int
                        })
        
        # If no rated suggestions found, try fallback patterns
        if not suggestions:
            for pattern in self.fallback_patterns:
                matches = pattern.findall(response)
                for original, improved in matches:
                    original_clean = original.strip()
                    improved_clean = improved.strip()
                    if original_clean and improved_clean and original_clean != improved_clean:
                        suggestions.append({
                            'original': original_clean,
                            'improved': improved_clean,
                            'rating': 5  # Default rating for suggestions without ratings
                        })
        
        # Filter out suggestions with rating below 4 and sort by rating (highest first)
        filtered_suggestions = [s for s in suggestions if s['rating'] >= 4]
        filtered_suggestions.sort(key=lambda x: x['rating'], reverse=True)
        
        logger.info(f"Parsed {len(suggestions)} total suggestions, {len(filtered_suggestions)} with rating >= 4")
        return filtered_suggestions

    def identify_high_priority_suggestions(self, suggestions):
        """Identify suggestions with rating > 7 for Pro model enhancement."""
        return [s for s in suggestions if s.get('rating', 0) > 7]

def create_direct_reviewer():
    """Create Gemini-based reviewer components."""
    gemini_client = GeminiClient()
    review_formatter = SimpleReviewFormatter()
    
    return gemini_client, review_formatter

def enhance_high_priority_suggestions(gemini_client, review_formatter, suggestions, original_text):
    """Enhance suggestions with rating > 7 using the Pro model."""
    if not suggestions:
        return suggestions
    
    high_priority = review_formatter.identify_high_priority_suggestions(suggestions)
    
    if not high_priority:
        logger.info("No high-priority suggestions found for Pro model enhancement")
        return suggestions
    
    logger.info(f"Enhancing {len(high_priority)} high-priority suggestions with Pro model")
    
    enhanced_suggestions = []
    for suggestion in suggestions:
        if suggestion.get('rating', 0) > 7:
            # Enhance with Pro model
            try:
                context = f"Part of larger text: {original_text[:200]}..." if len(original_text) > 200 else original_text
                enhancement_result = gemini_client.enhance_suggestion(
                    suggestion['original'], 
                    suggestion['improved'], 
                    context
                )
                
                if not enhancement_result['error']:
                    # Create enhanced suggestion
                    enhanced_suggestion = suggestion.copy()
                    enhanced_suggestion['improved'] = enhancement_result['enhanced_text']
                    enhanced_suggestion['enhanced'] = True
                    enhanced_suggestion['enhancement_model'] = enhancement_result['model']
                    enhanced_suggestions.append(enhanced_suggestion)
                    logger.info(f"Enhanced suggestion {suggestion.get('id', 'unknown')} with Pro model")
                else:
                    # Keep original on error
                    enhanced_suggestions.append(suggestion)
                    logger.warning(f"Failed to enhance suggestion {suggestion.get('id', 'unknown')}: {enhancement_result['error']}")
            except Exception as e:
                logger.error(f"Error enhancing suggestion {suggestion.get('id', 'unknown')}: {str(e)}")
                enhanced_suggestions.append(suggestion)
        else:
            # Keep suggestion as-is for rating <= 7
            enhanced_suggestions.append(suggestion)
    
    return enhanced_suggestions

def chunk_text(text, chunk_size_words=500, overlap_words=50):
    """Simple text chunking function."""
    if not text.strip():
        return []
    
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
    
    chunks = []
    current_word_idx = 0
    
    while current_word_idx < len(words):
        end_word_idx = min(current_word_idx + chunk_size_words, len(words))
        chunk_words = words[current_word_idx:end_word_idx]
        chunks.append(" ".join(chunk_words))
        
        step = chunk_size_words - overlap_words
        if step <= 0:
            step = max(1, chunk_size_words // 2)
        current_word_idx += step
    
    return chunks

@app.route('/api/feedback', methods=['POST'])
def get_feedback():
    """Get feedback for submitted text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Get request ID for tracking
        request_id = data.get('request_id', f'req_{int(time.time() * 1000)}')
        logger.info(f"Processing request {request_id} with text length {len(text)}")
        
        # Check if API key is available
        if not os.environ.get('GOOGLE_API_KEY'):
            return jsonify({'error': 'GOOGLE_API_KEY environment variable is required'}), 500
        
        # Create Gemini reviewer components
        gemini_client, review_formatter = create_direct_reviewer()
        
        # Process text directly
        logger.info(f"Processing text of {len(text)} characters")
        
        # Create optimized prompts for Gemini
        system_prompt = """Expert text reviewer. Analyze text and suggest improvements for clarity, grammar, and style.

Format: (change "original" -> "improved" [rating: X])
Rating 1-10: 1-3=minor, 4-6=moderate, 7-10=major improvements
Focus on: meaningful changes that enhance readability
Be: specific and actionable"""

        review_instructions = f"""Review this text:

{text}

Format: (change "original" -> "improved" [rating: X])"""
        
        # For shorter texts, process as single chunk (increased threshold for speed)
        if len(text.split()) <= 800:
            logger.info(f"Processing as single chunk ({len(text)} chars)")
            
            api_response = gemini_client.review_text_content(system_prompt, review_instructions)
            
            if api_response['error']:
                logger.error(f"Gemini API error: {api_response['error']}")
                return jsonify({'error': f'API error: {api_response["error"]}'}), 500
            
            all_suggestions = review_formatter.parse_suggestions(api_response['response'])
            
            # Add IDs to suggestions
            for sug_idx, sug in enumerate(all_suggestions):
                sug['id'] = f"s{sug_idx+1}"
                sug['chunk_index'] = 1
            
            # Enhance high-priority suggestions with Pro model
            all_suggestions = enhance_high_priority_suggestions(gemini_client, review_formatter, all_suggestions, text)
        else:
            # Chunk the text for longer content and process concurrently
            text_chunks = chunk_text(text, chunk_size_words=600, overlap_words=50)
            all_suggestions = []
            
            def process_chunk(chunk_data):
                i, chunk = chunk_data
                chunk_instructions = f"""Review this text:

{chunk}

Format: (change "original" -> "improved" [rating: X])"""
                
                api_response = gemini_client.review_text_content(system_prompt, chunk_instructions)
                
                if api_response['error']:
                    logger.error(f"API error in chunk {i+1}: {api_response['error']}")
                    return []
                
                chunk_suggestions = review_formatter.parse_suggestions(api_response['response'])
                
                # Add chunk info to suggestions
                for sug_idx, sug in enumerate(chunk_suggestions):
                    sug['id'] = f"c{i+1}_s{sug_idx+1}"
                    sug['chunk_index'] = i+1
                
                return chunk_suggestions
            
            # Process chunks concurrently (max 3 at a time to avoid rate limits)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                chunk_data = list(enumerate(text_chunks))
                logger.info(f"Processing {len(text_chunks)} chunks concurrently")
                
                futures = [executor.submit(process_chunk, chunk) for chunk in chunk_data]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_suggestions = future.result()
                        all_suggestions.extend(chunk_suggestions)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                
                # Enhance high-priority suggestions with Pro model
                all_suggestions = enhance_high_priority_suggestions(gemini_client, review_formatter, all_suggestions, text)
        
        response_data = {
            'suggestions': all_suggestions,
            'total_suggestions': len(all_suggestions),
            'status': 'success'
        }
        
        logger.info(f"Request {request_id} completed: {len(all_suggestions)} suggestions")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing feedback request: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'LLM Text Reviewer API'})

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'LLM Text Reviewer API',
        'version': '1.0',
        'endpoints': {
            '/api/feedback': 'POST - Get feedback for text',
            '/api/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    # Check for required environment variables
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: export GOOGLE_API_KEY=your_api_key_here")
        print("Get your API key at: https://aistudio.google.com/app/apikey")
    
    print("Starting Gemini Text Reviewer API server...")
    print("API will be available at: http://localhost:8000")
    print("Extension should connect to: http://localhost:8000/api/feedback")
    
    app.run(host='0.0.0.0', port=8000, debug=True)