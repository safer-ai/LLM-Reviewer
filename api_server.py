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

# Add the current directory to Python path to import reviewer
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key=None, model="gemini-2.5-flash-preview-05-20"):
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def review_text_content(self, system_prompt, user_prompt):
        """Get feedback from Gemini API."""
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.model.generate_content(full_prompt)
            
            return {
                'response': response.text,
                'model': self.model_name,
                'error': None
            }
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                'response': '',
                'model': self.model_name,
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

def create_direct_reviewer():
    """Create Gemini-based reviewer components."""
    gemini_client = GeminiClient()
    review_formatter = SimpleReviewFormatter()
    
    return gemini_client, review_formatter

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
        
        # Check if API key is available
        if not os.environ.get('GOOGLE_API_KEY'):
            return jsonify({'error': 'GOOGLE_API_KEY environment variable is required'}), 500
        
        # Create Gemini reviewer components
        gemini_client, review_formatter = create_direct_reviewer()
        
        # Process text directly
        logger.info(f"Processing text of {len(text)} characters")
        
        # Create prompts for Gemini
        system_prompt = """You are an expert text reviewer. Your task is to analyze the provided text and suggest specific improvements for clarity, grammar, style, and overall quality.

Please provide your suggestions in the following format:
- Use: (change "original text" -> "improved text" [rating: X]) for each suggestion
- Include a rating from 1-10 indicating how much this change improves the text:
  - 1-3: Minor improvements (typos, minor style tweaks)
  - 4-6: Moderate improvements (clarity, readability enhancements)
  - 7-10: Major improvements (significant clarity, grammar, or style fixes)
- Focus on meaningful improvements that enhance readability and correctness
- Be specific and actionable in your suggestions
- Only suggest changes that genuinely improve the text"""

        review_instructions = f"""Please review the following text for clarity, grammar, and style improvements:

Text to review:
{text}

Provide specific suggestions using the format: (change "original text" -> "improved text" [rating: X])
Remember to rate each suggestion from 1-10 based on its impact on improving the text."""
        
        # For shorter texts, process as single chunk
        if len(text.split()) <= 500:
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
        else:
            # Chunk the text for longer content
            text_chunks = chunk_text(text, chunk_size_words=500, overlap_words=50)
            all_suggestions = []
            
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
                
                chunk_instructions = f"""Please review the following text chunk for clarity, grammar, and style improvements:

Text chunk to review:
{chunk}

Provide specific suggestions using the format: (change "original text" -> "improved text" [rating: X])
Remember to rate each suggestion from 1-10 based on its impact on improving the text."""
                
                api_response = gemini_client.review_text_content(system_prompt, chunk_instructions)
                
                if api_response['error']:
                    logger.error(f"API error in chunk {i+1}: {api_response['error']}")
                    continue
                
                # Parse suggestions
                chunk_suggestions = review_formatter.parse_suggestions(api_response['response'])
                
                # Add chunk info to suggestions
                for sug_idx, sug in enumerate(chunk_suggestions):
                    sug['id'] = f"c{i+1}_s{sug_idx+1}"
                    sug['chunk_index'] = i+1
                
                all_suggestions.extend(chunk_suggestions)
        
        response_data = {
            'suggestions': all_suggestions,
            'total_suggestions': len(all_suggestions),
            'status': 'success'
        }
        
        logger.info(f"Found {len(all_suggestions)} suggestions")
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