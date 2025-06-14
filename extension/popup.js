class ExtensionApp {
    constructor() {
        this.suggestions = [];
        this.currentPage = 0;
        this.suggestionsPerPage = 3;
        this.init();
    }

    init() {
        this.bindEvents();
    }

    bindEvents() {
        const getFeedbackBtn = document.getElementById('getFeedbackBtn');
        const backBtn = document.getElementById('backBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const textInput = document.getElementById('textInput');

        getFeedbackBtn.addEventListener('click', () => this.handleGetFeedback());
        backBtn.addEventListener('click', () => this.showInputScreen());
        prevBtn.addEventListener('click', () => this.previousPage());
        nextBtn.addEventListener('click', () => this.nextPage());
        
        textInput.addEventListener('input', () => {
            getFeedbackBtn.disabled = textInput.value.trim().length === 0;
        });

        getFeedbackBtn.disabled = true;
    }

    async handleGetFeedback() {
        const textInput = document.getElementById('textInput');
        const text = textInput.value.trim();
        
        if (!text) {
            alert('Please enter some text to get feedback.');
            return;
        }

        this.showLoading(true);
        
        try {
            const suggestions = await this.getFeedbackFromAPI(text);
            this.suggestions = suggestions;
            this.currentPage = 0;
            this.showFeedbackScreen();
        } catch (error) {
            console.error('Error getting feedback:', error);
            alert('Failed to get feedback. Please check if the backend server is running.');
        } finally {
            this.showLoading(false);
        }
    }

    async getFeedbackFromAPI(text) {
        const response = await fetch('http://localhost:8000/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
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

        suggestionCounter.textContent = `${this.suggestions.length} suggestion${this.suggestions.length !== 1 ? 's' : ''} found (rating ≥ 4, sorted by importance)`;
        
        const startIndex = this.currentPage * this.suggestionsPerPage;
        const endIndex = Math.min(startIndex + this.suggestionsPerPage, this.suggestions.length);
        const currentSuggestions = this.suggestions.slice(startIndex, endIndex);
        
        suggestionRows.innerHTML = '';
        
        currentSuggestions.forEach(suggestion => {
            const row = document.createElement('div');
            row.className = 'table-row';
            
            const ratingClass = this.getRatingClass(suggestion.rating || 5);
            
            row.innerHTML = `
                <div class="cell rating-col">
                    <span class="rating-badge ${ratingClass}">${suggestion.rating || 5}</span>
                </div>
                <div class="cell original-col">${this.escapeHtml(suggestion.original)}</div>
                <div class="cell feedback-col">${this.escapeHtml(suggestion.improved)}</div>
            `;
            
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

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ExtensionApp();
});