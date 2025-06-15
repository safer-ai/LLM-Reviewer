class ExtensionApp {
    constructor() {
        this.suggestions = [];
        this.currentPage = 0;
        this.suggestionsPerPage = 3;
        this.isProcessing = false;
        this.requestQueue = [];
        this.init();
    }

    async init() {
        await this.loadStoredSuggestions();
        this.bindEvents();
        
        // Show suggestions screen if we have stored suggestions
        if (this.suggestions.length > 0) {
            this.showFeedbackScreen();
        }
    }

    bindEvents() {
        const getFeedbackBtn = document.getElementById('getFeedbackBtn');
        const seeSuggestionsBtn = document.getElementById('seeSuggestionsBtn');
        const backBtn = document.getElementById('backBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const clearAllBtn = document.getElementById('clearAllBtn');
        const textInput = document.getElementById('textInput');

        getFeedbackBtn.addEventListener('click', () => this.handleGetFeedback());
        seeSuggestionsBtn.addEventListener('click', () => this.showFeedbackScreen());
        backBtn.addEventListener('click', () => this.showInputScreen());
        prevBtn.addEventListener('click', () => this.previousPage());
        nextBtn.addEventListener('click', () => this.nextPage());
        clearAllBtn.addEventListener('click', () => this.clearAllSuggestions());
        
        textInput.addEventListener('input', () => {
            this.updateQueueStatus();
        });

        // Handle paste events - use plain text only
        textInput.addEventListener('paste', (e) => {
            e.preventDefault();
            const paste = e.clipboardData.getData('text/plain');
            
            if (paste) {
                document.execCommand('insertText', false, paste);
                this.updateQueueStatus();
            }
        });

        getFeedbackBtn.disabled = true;
        this.updateSeeSuggestionsButton();
    }

    async handleGetFeedback() {
        const textInput = document.getElementById('textInput');
        const text = this.getInputText().trim();
        
        if (!text) {
            alert('Please enter some text to get feedback.');
            return;
        }

        const requestData = {
            text: text,
            timestamp: new Date().toISOString(),
            id: `request_${Date.now()}`
        };

        if (this.isProcessing) {
            // Add to queue if currently processing
            this.requestQueue.push(requestData);
            await this.saveQueue();
            
            // Clear the input and show queue status
            textInput.innerHTML = '';
            document.getElementById('getFeedbackBtn').disabled = true;
            this.updateQueueStatus();
            
            return;
        }

        // Process immediately if not busy
        this.processRequest(requestData);
        
        // Clear the input
        textInput.innerHTML = '';
        document.getElementById('getFeedbackBtn').disabled = true;
    }

    async processRequest(requestData) {
        this.isProcessing = true;
        
        // Store the pending request
        await chrome.storage.local.set({ pendingRequest: requestData });
        this.showLoading(true);
        this.updateQueueStatus();
        
        try {
            const newSuggestions = await this.getFeedbackFromAPI(requestData.text, requestData.id);
            
            // Clear pending request
            await chrome.storage.local.remove(['pendingRequest']);
            
            // Add new suggestions to existing ones
            const timestamp = new Date().toISOString();
            const suggestionsWithMeta = newSuggestions.map((suggestion, index) => ({
                ...suggestion,
                id: `${timestamp}_${index}`,
                timestamp: timestamp
            }));
            
            // Remove duplicates based on original and improved text
            const combinedSuggestions = [...suggestionsWithMeta, ...this.suggestions];
            this.suggestions = this.removeDuplicates(combinedSuggestions);
            await this.saveSuggestions();
            
            // Update the See Suggestions button
            this.updateSeeSuggestionsButton();
            
            this.currentPage = 0;
            this.showFeedbackScreen();
            
        } catch (error) {
            console.error('Error getting feedback:', error);
            alert('Failed to get feedback. Please check if the backend server is running.');
            // Don't clear pending request on error, so it can be resumed
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
            this.updateQueueStatus();
            
            // Process next request in queue
            if (this.requestQueue.length > 0) {
                setTimeout(() => this.processNextRequest(), 1000);
            }
        }
    }

    async processNextRequest() {
        if (this.requestQueue.length > 0 && !this.isProcessing) {
            const nextRequest = this.requestQueue.shift();
            await this.saveQueue();
            this.processRequest(nextRequest);
        }
    }

    async resumePendingRequest(requestData) {
        console.log('Resuming pending request:', requestData);
        this.processRequest(requestData);
    }

    async getFeedbackFromAPI(text, requestId = null) {
        const response = await fetch('http://localhost:8000/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text,
                request_id: requestId || `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.suggestions || [];
    }

    showLoading(show) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const getFeedbackBtn = document.getElementById('getFeedbackBtn');
        
        if (show) {
            loadingIndicator.classList.remove('hidden');
            getFeedbackBtn.disabled = true;
        } else {
            loadingIndicator.classList.add('hidden');
            getFeedbackBtn.disabled = false;
        }
    }

    showInputScreen() {
        document.getElementById('inputScreen').classList.remove('hidden');
        document.getElementById('feedbackScreen').classList.add('hidden');
    }

    showFeedbackScreen() {
        document.getElementById('inputScreen').classList.add('hidden');
        document.getElementById('feedbackScreen').classList.remove('hidden');
        this.renderFeedback();
    }

    renderFeedback() {
        const suggestionCounter = document.getElementById('suggestionCounter');
        const suggestionRows = document.getElementById('suggestionRows');
        
        if (this.suggestions.length === 0) {
            suggestionCounter.textContent = 'No suggestions found.';
            suggestionRows.innerHTML = '<div class="table-row"><div class="cell" style="text-align: center; color: #6c757d;">No feedback available for this text.</div></div>';
            this.updateNavigation();
            return;
        }

        suggestionCounter.textContent = `${this.suggestions.length} suggestion${this.suggestions.length !== 1 ? 's' : ''} in backlog`;
        
        const startIndex = this.currentPage * this.suggestionsPerPage;
        const endIndex = Math.min(startIndex + this.suggestionsPerPage, this.suggestions.length);
        const currentSuggestions = this.suggestions.slice(startIndex, endIndex);
        
        suggestionRows.innerHTML = '';
        
        currentSuggestions.forEach((suggestion, index) => {
            const row = document.createElement('div');
            row.className = `table-row ${suggestion.enhanced ? 'enhanced-suggestion' : ''}`;
            
            const ratingClass = this.getRatingClass(suggestion.rating || 5);
            const globalIndex = startIndex + index;
            
            const ratingCell = document.createElement('div');
            ratingCell.className = 'cell rating-col';
            
            const ratingContent = `<span class="rating-badge ${ratingClass}">${suggestion.rating || 5}</span>`;
            const enhancedBadge = suggestion.enhanced ? '<span class="enhanced-badge">Pro</span>' : '';
            ratingCell.innerHTML = ratingContent + enhancedBadge;
            
            const originalCell = document.createElement('div');
            originalCell.className = 'cell original-col';
            originalCell.innerHTML = this.formatTextForDisplay(suggestion.original);
            
            const improvedCell = document.createElement('div');
            improvedCell.className = 'cell feedback-col';
            improvedCell.innerHTML = this.formatTextForDisplay(suggestion.improved);
            
            const actionCell = document.createElement('div');
            actionCell.className = 'cell action-col';
            
            const doneBtn = document.createElement('button');
            doneBtn.className = 'done-btn';
            doneBtn.textContent = "I'm done";
            doneBtn.addEventListener('click', () => this.markSuggestionDone(globalIndex));
            
            actionCell.appendChild(doneBtn);
            row.appendChild(ratingCell);
            row.appendChild(originalCell);
            row.appendChild(improvedCell);
            row.appendChild(actionCell);
            
            suggestionRows.appendChild(row);
        });
        
        this.updateNavigation();
    }

    updateNavigation() {
        const totalPages = Math.ceil(this.suggestions.length / this.suggestionsPerPage);
        const currentPageSpan = document.getElementById('currentPage');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        
        if (totalPages <= 1) {
            document.getElementById('navigationControls').style.display = 'none';
            return;
        }
        
        document.getElementById('navigationControls').style.display = 'flex';
        currentPageSpan.textContent = `${this.currentPage + 1} / ${totalPages}`;
        prevBtn.disabled = this.currentPage === 0;
        nextBtn.disabled = this.currentPage >= totalPages - 1;
    }

    previousPage() {
        if (this.currentPage > 0) {
            this.currentPage--;
            this.renderFeedback();
        }
    }

    nextPage() {
        const totalPages = Math.ceil(this.suggestions.length / this.suggestionsPerPage);
        if (this.currentPage < totalPages - 1) {
            this.currentPage++;
            this.renderFeedback();
        }
    }

    getRatingClass(rating) {
        if (rating >= 7) return 'rating-high';
        if (rating >= 4) return 'rating-medium';
        return 'rating-low';
    }

    async loadStoredSuggestions() {
        try {
            const result = await chrome.storage.local.get(['suggestions', 'pendingRequest', 'requestQueue']);
            if (result.suggestions) {
                this.suggestions = result.suggestions;
            }
            
            if (result.requestQueue) {
                this.requestQueue = result.requestQueue;
            }
            
            // Check if there's a pending request
            if (result.pendingRequest) {
                this.isProcessing = true;
                this.resumePendingRequest(result.pendingRequest);
            } else if (this.requestQueue.length > 0) {
                // Process queued requests
                this.processNextRequest();
            }
        } catch (error) {
            console.error('Error loading stored suggestions:', error);
        }
    }

    removeDuplicates(suggestions) {
        const seen = new Set();
        return suggestions.filter(suggestion => {
            const key = `${suggestion.original.toLowerCase().trim()}|${suggestion.improved.toLowerCase().trim()}`;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }

    async saveSuggestions() {
        try {
            await chrome.storage.local.set({ suggestions: this.suggestions });
        } catch (error) {
            console.error('Error saving suggestions:', error);
        }
    }

    async saveQueue() {
        try {
            await chrome.storage.local.set({ requestQueue: this.requestQueue });
        } catch (error) {
            console.error('Error saving queue:', error);
        }
    }

    updateQueueStatus() {
        const getFeedbackBtn = document.getElementById('getFeedbackBtn');
        const textInput = document.getElementById('textInput');
        
        if (this.isProcessing) {
            getFeedbackBtn.textContent = `Processing... (${this.requestQueue.length} queued)`;
            getFeedbackBtn.disabled = false; // Allow queueing more requests
        } else {
            getFeedbackBtn.textContent = 'Get Feedback';
            getFeedbackBtn.disabled = this.getInputText().trim().length === 0;
        }
        
        this.updateSeeSuggestionsButton();
    }

    updateSeeSuggestionsButton() {
        const seeSuggestionsBtn = document.getElementById('seeSuggestionsBtn');
        
        if (this.suggestions.length > 0) {
            seeSuggestionsBtn.classList.remove('hidden');
            seeSuggestionsBtn.textContent = `See Suggestions (${this.suggestions.length})`;
        } else {
            seeSuggestionsBtn.classList.add('hidden');
        }
    }

    async markSuggestionDone(index) {
        if (index >= 0 && index < this.suggestions.length) {
            // Remove the suggestion from the array
            this.suggestions.splice(index, 1);
            await this.saveSuggestions();
            
            // Adjust current page if needed
            const maxPage = Math.max(0, Math.ceil(this.suggestions.length / this.suggestionsPerPage) - 1);
            if (this.currentPage > maxPage) {
                this.currentPage = maxPage;
            }
            
            // Update the See Suggestions button
            this.updateSeeSuggestionsButton();
            
            // If no suggestions left, go back to input screen
            if (this.suggestions.length === 0) {
                this.showInputScreen();
            } else {
                this.renderFeedback();
            }
        }
    }

    async clearAllSuggestions() {
        if (confirm('Are you sure you want to clear all suggestions? This cannot be undone.')) {
            this.suggestions = [];
            await this.saveSuggestions();
            this.updateSeeSuggestionsButton();
            this.showInputScreen();
        }
    }

    getInputText() {
        const textInput = document.getElementById('textInput');
        return textInput.innerText || textInput.textContent || '';
    }

    getInputHTML() {
        const textInput = document.getElementById('textInput');
        return textInput.innerHTML || '';
    }


    formatTextForDisplay(text) {
        // Convert plain text line breaks to HTML line breaks
        if (typeof text === 'string') {
            return text
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>')
                .replace(/^(.*)$/, '<p>$1</p>')
                .replace(/<p><\/p>/g, '<p>&nbsp;</p>');
        }
        return text;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Make app globally accessible for onclick handlers
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new ExtensionApp();
});