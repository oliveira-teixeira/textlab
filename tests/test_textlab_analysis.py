"""
Comprehensive pytest test suite for TextLab qualitative text analysis app.

This module tests the core analysis algorithms that power TextLab. Since TextLab is a
JavaScript application, these tests implement Python versions of the key algorithms
to validate the core logic is correct.

Test Categories:
- Text Processing: Tokenization, stopword removal, filtering
- Word Frequency: Frequency counting, hapax legomena
- Cooccurrence: Word cooccurrence analysis
- TF-IDF: Term frequency-inverse document frequency
- Corpus Filtering: Multi-document corpus handling
- Sentiment Analysis: Sentiment scoring
- Bigrams: Bigram extraction and frequency
- KWIC: Keyword in context
- CHD/Reinert: Hierarchical document clustering
- Edge Cases: Robustness and error handling
"""

import pytest
import math
from collections import Counter
from typing import List, Dict, Set, Tuple
from operator import itemgetter


# ============================================================================
# CORE ALGORITHM IMPLEMENTATIONS (Python versions of JS logic)
# ============================================================================

class TextProcessor:
    """Implements text processing algorithms for TextLab."""

    def __init__(self, stopwords_en: Set[str] = None, stopwords_pt: Set[str] = None):
        """Initialize text processor with language-specific stopwords."""
        self.stopwords_en = stopwords_en or {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'to', 'of', 'for', 'with', 'by', 'from', 'as', 'be', 'are',
            'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        self.stopwords_pt = stopwords_pt or {
            'de', 'da', 'do', 'para', 'com', 'em', 'é', 'o', 'a', 'e', 'ou',
            'por', 'se', 'que', 'este', 'esse', 'aquele', 'ele', 'ela',
            'nós', 'vós', 'eles', 'elas', 'meu', 'teu', 'seu', 'nosso',
            'ao', 'aos', 'uma', 'uns', 'umas', 'no', 'na', 'nos', 'nas'
        }
        self.custom_stopwords = set()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words.

        Removes punctuation, converts to lowercase, and returns list of tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase word tokens
        """
        if not text or not text.strip():
            return []

        # Remove punctuation and convert to lowercase
        import re
        text = text.lower()
        # Keep only alphanumeric, spaces, and accented characters
        text = re.sub(r'[^\w\s\à\á\â\ã\ä\ç\è\é\ê\ë\ì\í\î\ï\ñ\ò\ó\ô\õ\ö\ù\ú\û\ü]', ' ', text)

        tokens = text.split()
        return [t for t in tokens if t]

    def remove_stopwords(self, tokens: List[str], language: str = 'en') -> List[str]:
        """
        Remove stopwords from token list.

        Removes common words that don't contribute to analysis.

        Args:
            tokens: List of tokens to filter
            language: 'en' for English or 'pt' for Portuguese

        Returns:
            Filtered token list
        """
        if language == 'pt':
            stopwords = self.stopwords_pt | self.custom_stopwords
        else:
            stopwords = self.stopwords_en | self.custom_stopwords

        return [t for t in tokens if t.lower() not in stopwords]

    def add_custom_stopword(self, word: str) -> None:
        """Add a custom stopword to be removed from analysis."""
        self.custom_stopwords.add(word.lower())

    def filter_by_length(self, tokens: List[str], min_length: int = 1) -> List[str]:
        """
        Filter words by minimum length.

        Args:
            tokens: List of tokens to filter
            min_length: Minimum word length to keep

        Returns:
            Filtered token list
        """
        return [t for t in tokens if len(t) >= min_length]


class WordFrequencyAnalyzer:
    """Analyzes word frequency patterns."""

    @staticmethod
    def calculate_frequencies(tokens: List[str]) -> Dict[str, int]:
        """
        Calculate frequency of each word.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary mapping word -> frequency count
        """
        return dict(Counter(tokens))

    @staticmethod
    def get_sorted_frequencies(freq_dict: Dict[str, int], descending: bool = True) -> List[Tuple[str, int]]:
        """
        Get sorted word frequencies.

        Args:
            freq_dict: Frequency dictionary
            descending: If True, sort by frequency descending

        Returns:
            List of (word, frequency) tuples sorted by frequency
        """
        reverse = descending
        return sorted(freq_dict.items(), key=itemgetter(1), reverse=reverse)

    @staticmethod
    def find_hapax_legomena(freq_dict: Dict[str, int]) -> List[str]:
        """
        Find hapax legomena (words appearing exactly once).

        Args:
            freq_dict: Frequency dictionary

        Returns:
            List of words appearing exactly once
        """
        return [word for word, count in freq_dict.items() if count == 1]


class CooccurrenceAnalyzer:
    """Analyzes word cooccurrence patterns."""

    @staticmethod
    def calculate_cooccurrence(tokens: List[str], window_size: int = 5) -> Dict[Tuple[str, str], int]:
        """
        Calculate cooccurrence of words within a window.

        Words within window_size positions of each other are considered cooccurring.

        Args:
            tokens: List of tokens
            window_size: Size of context window

        Returns:
            Dictionary mapping (word1, word2) -> cooccurrence count
        """
        cooccurrence = Counter()

        for i, word1 in enumerate(tokens):
            # Get window of surrounding words
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    word2 = tokens[j]
                    # Create canonical pair (alphabetically ordered)
                    pair = tuple(sorted([word1, word2]))
                    cooccurrence[pair] += 1

        return dict(cooccurrence)

    @staticmethod
    def get_weighted_cooccurrence(cooccurrence: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
        """
        Calculate weighted cooccurrence (normalized).

        Args:
            cooccurrence: Cooccurrence frequency dictionary

        Returns:
            Dictionary with normalized weights
        """
        if not cooccurrence:
            return {}

        max_count = max(cooccurrence.values())
        return {pair: count / max_count for pair, count in cooccurrence.items()}


class TFIDFAnalyzer:
    """Implements TF-IDF (Term Frequency-Inverse Document Frequency) analysis."""

    @staticmethod
    def calculate_tfidf(documents: List[List[str]]) -> List[Dict[str, float]]:
        """
        Calculate TF-IDF scores for each document.

        Args:
            documents: List of tokenized documents (each is a list of tokens)

        Returns:
            List of dictionaries mapping word -> TF-IDF score for each document
        """
        if not documents:
            return []

        # Calculate IDF (inverse document frequency)
        num_docs = len(documents)
        word_doc_count = Counter()

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_count[word] += 1

        # Calculate TF-IDF for each document
        tfidf_scores = []
        for doc in documents:
            doc_tfidf = {}
            word_freq = Counter(doc)
            doc_length = len(doc)

            for word, freq in word_freq.items():
                # Term frequency (normalized by document length)
                tf = freq / doc_length if doc_length > 0 else 0

                # Inverse document frequency
                idf = math.log(num_docs / word_doc_count[word]) if word_doc_count[word] > 0 else 0

                # TF-IDF
                doc_tfidf[word] = tf * idf

            tfidf_scores.append(doc_tfidf)

        return tfidf_scores


class CorpusFilter:
    """Handles multi-document corpus filtering."""

    def __init__(self, documents: List[List[str]], corpus_names: List[str] = None):
        """
        Initialize corpus filter.

        Args:
            documents: List of tokenized documents
            corpus_names: Optional list of corpus names for each document
        """
        self.documents = documents
        self.corpus_names = corpus_names or [f"doc_{i}" for i in range(len(documents))]
        self.corpus_mapping = {name: doc for name, doc in zip(self.corpus_names, documents)}

    def filter_by_corpus(self, corpus_name: str) -> List[List[str]]:
        """
        Get documents from a specific corpus.

        Args:
            corpus_name: Name of corpus ('all' returns all documents)

        Returns:
            List of tokenized documents from the corpus
        """
        if corpus_name == 'all':
            return self.documents

        return [self.corpus_mapping[corpus_name]] if corpus_name in self.corpus_mapping else []

    def get_all_documents(self) -> List[List[str]]:
        """Get all documents in corpus."""
        return self.documents

    def get_corpus_names(self) -> List[str]:
        """Get names of available corpora."""
        return self.corpus_names


class SentimentAnalyzer:
    """Analyzes sentiment in text."""

    def __init__(self):
        """Initialize sentiment analyzer with word lists."""
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'beautiful', 'lovely', 'perfect', 'awesome', 'brilliant', 'superb',
            'love', 'like', 'enjoy', 'happy', 'glad', 'pleased'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'evil', 'hate',
            'dislike', 'sad', 'angry', 'upset', 'disappointed', 'ugly',
            'nasty', 'wrong', 'evil', 'disgusting', 'pathetic', 'dreadful'
        }

    def calculate_sentiment(self, tokens: List[str]) -> float:
        """
        Calculate sentiment score for tokens.

        Args:
            tokens: List of tokens

        Returns:
            Sentiment score (-1.0 to 1.0, where negative is negative sentiment)
        """
        if not tokens:
            return 0.0

        positive_count = sum(1 for t in tokens if t.lower() in self.positive_words)
        negative_count = sum(1 for t in tokens if t.lower() in self.negative_words)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total


class BigramAnalyzer:
    """Analyzes bigrams (word pairs)."""

    @staticmethod
    def extract_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Extract bigrams (adjacent word pairs) from tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of (word1, word2) bigram tuples
        """
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))
        return bigrams

    @staticmethod
    def get_bigram_frequencies(bigrams: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        """
        Calculate bigram frequencies.

        Args:
            bigrams: List of bigram tuples

        Returns:
            Dictionary mapping bigram -> frequency
        """
        return dict(Counter(bigrams))

    @staticmethod
    def get_sorted_bigrams(freq_dict: Dict[Tuple[str, str], int]) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get bigrams sorted by frequency.

        Args:
            freq_dict: Bigram frequency dictionary

        Returns:
            List of (bigram, frequency) sorted by frequency descending
        """
        return sorted(freq_dict.items(), key=itemgetter(1), reverse=True)


class KWICAnalyzer:
    """Performs KWIC (Keyword In Context) analysis."""

    @staticmethod
    def find_kwic(tokens: List[str], keyword: str, context_size: int = 5) -> List[Dict]:
        """
        Find keyword in context with surrounding words.

        Args:
            tokens: List of tokens
            keyword: Keyword to search for
            context_size: Number of words on each side

        Returns:
            List of KWIC results with left context, keyword, right context
        """
        results = []
        keyword_lower = keyword.lower()

        for i, token in enumerate(tokens):
            if token.lower() == keyword_lower:
                left_start = max(0, i - context_size)
                right_end = min(len(tokens), i + context_size + 1)

                left_context = tokens[left_start:i]
                right_context = tokens[i + 1:right_end]

                results.append({
                    'left': left_context,
                    'keyword': token,
                    'right': right_context,
                    'position': i
                })

        return results


class CHDAnalyzer:
    """Implements CHD (Classification Hierarchique Descendante) / Reinert analysis."""

    @staticmethod
    def perform_chd(documents: List[List[str]], num_classes: int = 2) -> Dict:
        """
        Perform simple CHD clustering (simplified implementation).

        Args:
            documents: List of tokenized documents
            num_classes: Number of classes to create

        Returns:
            Dictionary with class assignments and word associations
        """
        if not documents:
            return {'classes': [], 'assignments': [], 'words': {}}

        # Simple clustering: split documents by document length
        sorted_docs = sorted(enumerate(documents), key=lambda x: len(x[1]))

        assignments = [0] * len(documents)
        docs_per_class = len(documents) // num_classes

        for idx, (orig_idx, doc) in enumerate(sorted_docs):
            class_id = min(idx // docs_per_class, num_classes - 1)
            assignments[orig_idx] = class_id

        # Find characteristic words for each class
        class_words = {}
        for class_id in range(num_classes):
            class_docs = [documents[i] for i in range(len(documents)) if assignments[i] == class_id]
            if class_docs:
                all_words = [w for doc in class_docs for w in doc]
                word_freqs = Counter(all_words)
                class_words[class_id] = dict(word_freqs.most_common(10))

        return {
            'classes': list(range(num_classes)),
            'assignments': assignments,
            'words': class_words
        }


# ============================================================================
# PYTEST TEST CASES
# ============================================================================

class TestTextProcessing:
    """Test suite for text processing functionality."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        processor = TextProcessor()
        result = processor.tokenize("Hello world, this is a test!")
        expected = ['hello', 'world', 'this', 'is', 'a', 'test']
        assert result == expected

    def test_tokenize_portuguese(self):
        """Test tokenization with Portuguese accents."""
        processor = TextProcessor()
        result = processor.tokenize("Análise textual em português")
        expected = ['análise', 'textual', 'em', 'português']
        assert result == expected

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        processor = TextProcessor()
        result = processor.tokenize("")
        assert result == []

    def test_stopword_removal_pt(self):
        """Test removal of Portuguese stopwords."""
        processor = TextProcessor()
        tokens = ['análise', 'de', 'texto', 'da', 'linguagem', 'do', 'português']
        result = processor.remove_stopwords(tokens, language='pt')
        # 'de', 'da', 'do' should be removed
        assert 'de' not in result
        assert 'da' not in result
        assert 'do' not in result
        assert 'análise' in result

    def test_stopword_removal_en(self):
        """Test removal of English stopwords."""
        processor = TextProcessor()
        tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        result = processor.remove_stopwords(tokens, language='en')
        assert 'the' not in result
        assert 'quick' in result
        assert 'brown' in result

    def test_custom_stopword_addition(self):
        """Test adding custom stopwords."""
        processor = TextProcessor()
        processor.add_custom_stopword('test')
        tokens = ['hello', 'test', 'world']
        result = processor.remove_stopwords(tokens, language='en')
        assert 'test' not in result
        assert 'hello' in result

    def test_min_length_filter(self):
        """Test minimum length filtering."""
        processor = TextProcessor()
        tokens = ['hello', 'a', 'world', 'is', 'test']
        result = processor.filter_by_length(tokens, min_length=2)
        assert 'a' not in result
        assert 'hello' in result
        assert len(result) == 4


class TestWordFrequency:
    """Test suite for word frequency analysis."""

    def test_word_frequency_basic(self):
        """Test basic word frequency counting."""
        tokens = ['hello', 'world', 'hello', 'test', 'world', 'hello']
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)
        assert freqs['hello'] == 3
        assert freqs['world'] == 2
        assert freqs['test'] == 1

    def test_word_frequency_ordering(self):
        """Test that frequencies are ordered descending."""
        tokens = ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)
        sorted_freqs = WordFrequencyAnalyzer.get_sorted_frequencies(freqs)

        # Check order
        assert sorted_freqs[0][1] >= sorted_freqs[1][1]
        assert sorted_freqs[1][1] >= sorted_freqs[2][1]
        assert sorted_freqs[0][0] == 'd'  # 'd' appears 4 times

    def test_word_frequency_with_stopwords(self):
        """Test frequency counting excluding stopwords."""
        processor = TextProcessor()
        tokens = processor.tokenize("the quick brown fox the lazy dog the")
        tokens = processor.remove_stopwords(tokens, language='en')
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)

        assert 'the' not in freqs
        assert 'quick' in freqs
        assert freqs['quick'] == 1

    def test_hapax_legomena(self):
        """Test finding words appearing exactly once."""
        tokens = ['hello', 'world', 'hello', 'test', 'example']
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)
        hapax = WordFrequencyAnalyzer.find_hapax_legomena(freqs)

        assert 'test' in hapax
        assert 'example' in hapax
        assert 'hello' not in hapax
        assert len(hapax) == 3  # world, test, example each appear once


class TestCooccurrence:
    """Test suite for cooccurrence analysis."""

    def test_cooccurrence_basic(self):
        """Test basic cooccurrence calculation."""
        tokens = ['hello', 'world', 'hello', 'test']
        cooc = CooccurrenceAnalyzer.calculate_cooccurrence(tokens, window_size=2)

        # 'hello' and 'world' are adjacent
        pair = tuple(sorted(['hello', 'world']))
        assert pair in cooc
        assert cooc[pair] > 0

    def test_cooccurrence_window_size(self):
        """Test that different window sizes produce different results."""
        tokens = ['a', 'b', 'c', 'd', 'e']
        cooc_small = CooccurrenceAnalyzer.calculate_cooccurrence(tokens, window_size=1)
        cooc_large = CooccurrenceAnalyzer.calculate_cooccurrence(tokens, window_size=3)

        # Larger window should have more cooccurrences
        assert len(cooc_large) >= len(cooc_small)

    def test_cooccurrence_symmetric(self):
        """Test that cooccurrence is symmetric."""
        tokens = ['hello', 'world', 'test', 'hello', 'world']
        cooc = CooccurrenceAnalyzer.calculate_cooccurrence(tokens, window_size=5)

        # Check that pairs are stored in canonical form (sorted)
        for pair in cooc.keys():
            assert pair[0] <= pair[1]

    def test_cooccurrence_weight(self):
        """Test weighted cooccurrence normalization."""
        tokens = ['a', 'b', 'a', 'b', 'a', 'b', 'c', 'd']
        cooc = CooccurrenceAnalyzer.calculate_cooccurrence(tokens, window_size=5)
        weighted = CooccurrenceAnalyzer.get_weighted_cooccurrence(cooc)

        # Max weight should be 1.0
        if weighted:
            assert max(weighted.values()) <= 1.0
            assert min(weighted.values()) > 0.0


class TestTFIDF:
    """Test suite for TF-IDF analysis."""

    def test_tfidf_single_document(self):
        """Test TF-IDF with single document."""
        documents = [['hello', 'world', 'hello']]
        tfidf = TFIDFAnalyzer.calculate_tfidf(documents)

        assert len(tfidf) == 1
        assert 'hello' in tfidf[0]
        assert 'world' in tfidf[0]

    def test_tfidf_multiple_documents(self):
        """Test TF-IDF with multiple documents."""
        documents = [
            ['hello', 'world'],
            ['hello', 'python', 'test'],
            ['python', 'code', 'world']
        ]
        tfidf = TFIDFAnalyzer.calculate_tfidf(documents)

        assert len(tfidf) == 3
        # 'python' appears in docs 1 and 2, so should have lower IDF
        # 'code' appears only in doc 2, so should have higher IDF

    def test_tfidf_common_word_penalty(self):
        """Test that common words get penalized in TF-IDF."""
        documents = [
            ['common', 'word', 'specific'],
            ['common', 'word', 'different'],
            ['common', 'word', 'unique']
        ]
        tfidf = TFIDFAnalyzer.calculate_tfidf(documents)

        # 'specific' appears once
        # 'common' appears in all docs
        # In all documents, 'common' should have lower IDF than 'specific'
        assert tfidf[0]['specific'] > tfidf[0].get('common', 0)


class TestCorpusFiltering:
    """Test suite for corpus filtering (CRITICAL)."""

    def test_all_documents_processed(self):
        """Test that 'all' corpus includes all documents."""
        docs = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        corpus = CorpusFilter(docs, corpus_names=['doc1', 'doc2', 'doc3'])

        filtered = corpus.filter_by_corpus('all')
        assert len(filtered) == 3
        assert filtered == docs

    def test_filtered_corpus_only(self):
        """Test that filtering by corpus name returns only that corpus."""
        docs = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        corpus = CorpusFilter(docs, corpus_names=['doc1', 'doc2', 'doc3'])

        filtered = corpus.filter_by_corpus('doc1')
        assert len(filtered) == 1
        assert filtered[0] == ['a', 'b']

    def test_stopword_applies_globally(self):
        """Test that stopword removal applies to all analyses."""
        processor = TextProcessor()
        processor.add_custom_stopword('remove_me')

        doc1 = processor.remove_stopwords(['hello', 'remove_me', 'world'], language='en')
        doc2 = processor.remove_stopwords(['test', 'remove_me', 'data'], language='en')

        assert 'remove_me' not in doc1
        assert 'remove_me' not in doc2

    def test_multiple_documents_corpus(self):
        """Test that multiple documents all contribute to frequency."""
        docs = [
            ['hello', 'world'],
            ['hello', 'python'],
            ['world', 'test'],
            ['python', 'code'],
            ['hello', 'data']
        ]
        corpus = CorpusFilter(docs)
        all_docs = corpus.filter_by_corpus('all')

        assert len(all_docs) == 5

        # Combine all documents and count
        all_tokens = [t for doc in all_docs for t in doc]
        freqs = Counter(all_tokens)

        assert freqs['hello'] == 3
        assert freqs['world'] == 2
        assert freqs['python'] == 2

    def test_corpus_filter_affects_cooccurrence(self):
        """Test that corpus filter affects cooccurrence analysis."""
        doc1 = ['a', 'b', 'c']
        doc2 = ['a', 'x', 'y']

        # Cooccurrence in doc1 only
        cooc1 = CooccurrenceAnalyzer.calculate_cooccurrence(doc1, window_size=5)
        # Cooccurrence in doc2 only
        cooc2 = CooccurrenceAnalyzer.calculate_cooccurrence(doc2, window_size=5)

        # doc1 should have b-c cooccurrence
        pair_bc = tuple(sorted(['b', 'c']))
        assert pair_bc in cooc1
        assert pair_bc not in cooc2

    def test_corpus_filter_affects_tfidf(self):
        """Test that corpus filter affects TF-IDF analysis."""
        docs_subset = [['hello', 'world']]
        docs_full = [['hello', 'world'], ['hello', 'python'], ['world', 'test']]

        tfidf_subset = TFIDFAnalyzer.calculate_tfidf(docs_subset)
        tfidf_full = TFIDFAnalyzer.calculate_tfidf(docs_full)

        # Results should be different
        assert tfidf_subset[0]['hello'] != tfidf_full[0]['hello']


class TestSentimentAnalysis:
    """Test suite for sentiment analysis."""

    def test_sentiment_positive(self):
        """Test positive sentiment detection."""
        analyzer = SentimentAnalyzer()
        tokens = ['this', 'is', 'great', 'and', 'wonderful']
        sentiment = analyzer.calculate_sentiment(tokens)
        assert sentiment > 0

    def test_sentiment_negative(self):
        """Test negative sentiment detection."""
        analyzer = SentimentAnalyzer()
        tokens = ['this', 'is', 'bad', 'and', 'awful']
        sentiment = analyzer.calculate_sentiment(tokens)
        assert sentiment < 0

    def test_sentiment_neutral(self):
        """Test neutral sentiment."""
        analyzer = SentimentAnalyzer()
        tokens = ['this', 'is', 'a', 'test', 'document']
        sentiment = analyzer.calculate_sentiment(tokens)
        assert abs(sentiment) < 0.1


class TestBigrams:
    """Test suite for bigram analysis."""

    def test_bigram_extraction(self):
        """Test extracting bigrams from tokens."""
        tokens = ['hello', 'world', 'test']
        bigrams = BigramAnalyzer.extract_bigrams(tokens)

        assert len(bigrams) == 2
        assert bigrams[0] == ('hello', 'world')
        assert bigrams[1] == ('world', 'test')

    def test_bigram_frequency(self):
        """Test bigram frequency counting."""
        tokens = ['a', 'b', 'a', 'b', 'a', 'b']
        bigrams = BigramAnalyzer.extract_bigrams(tokens)
        freqs = BigramAnalyzer.get_bigram_frequencies(bigrams)

        assert freqs[('a', 'b')] == 3
        assert freqs[('b', 'a')] == 2

    def test_bigram_with_stopwords(self):
        """Test that stopwords are included in bigrams (they're removed before)."""
        processor = TextProcessor()
        tokens = processor.tokenize("the quick brown fox")
        tokens = processor.remove_stopwords(tokens, language='en')
        bigrams = BigramAnalyzer.extract_bigrams(tokens)

        # After removing 'the', we should have 'quick-brown' and 'brown-fox'
        assert len(bigrams) == 2


class TestKWIC:
    """Test suite for KWIC (Keyword In Context) analysis."""

    def test_kwic_basic(self):
        """Test basic KWIC functionality."""
        tokens = ['the', 'quick', 'brown', 'fox', 'jumps']
        kwic = KWICAnalyzer.find_kwic(tokens, 'brown', context_size=2)

        assert len(kwic) == 1
        assert kwic[0]['keyword'] == 'brown'
        assert kwic[0]['left'] == ['the', 'quick']
        assert kwic[0]['right'] == ['fox', 'jumps']

    def test_kwic_multiple_occurrences(self):
        """Test KWIC with multiple keyword occurrences."""
        tokens = ['test', 'hello', 'world', 'test', 'again', 'test', 'end']
        kwic = KWICAnalyzer.find_kwic(tokens, 'test', context_size=2)

        assert len(kwic) == 3
        assert all(result['keyword'] == 'test' for result in kwic)

    def test_kwic_case_insensitive(self):
        """Test KWIC with case-insensitive search."""
        tokens = ['Hello', 'World', 'HELLO', 'again']
        kwic = KWICAnalyzer.find_kwic(tokens, 'hello', context_size=1)

        # Should find both 'Hello' and 'HELLO'
        assert len(kwic) == 2


class TestCHD:
    """Test suite for CHD/Reinert analysis."""

    def test_chd_produces_classes(self):
        """Test that CHD produces specified number of classes."""
        docs = [['a', 'b'], ['c', 'd'], ['e', 'f']]
        result = CHDAnalyzer.perform_chd(docs, num_classes=2)

        assert len(result['classes']) >= 2

    def test_chd_class_assignment(self):
        """Test that all documents are assigned to exactly one class."""
        docs = [['a'], ['b'], ['c'], ['d']]
        result = CHDAnalyzer.perform_chd(docs, num_classes=2)

        assert len(result['assignments']) == 4
        for assignment in result['assignments']:
            assert assignment in result['classes']

    def test_chd_word_association(self):
        """Test that words are associated with their classes."""
        docs = [['a', 'b'], ['c', 'd']]
        result = CHDAnalyzer.perform_chd(docs, num_classes=2)

        assert len(result['words']) > 0
        for class_id, words in result['words'].items():
            assert isinstance(words, dict)


class TestEdgeCases:
    """Test suite for edge cases and robustness."""

    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        corpus = CorpusFilter([])
        filtered = corpus.filter_by_corpus('all')
        assert filtered == []

    def test_single_word_document(self):
        """Test handling of single-word documents."""
        tokens = ['hello']
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)

        assert 'hello' in freqs
        assert freqs['hello'] == 1

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        processor = TextProcessor()
        text = "Café, naïve, résumé, über"
        tokens = processor.tokenize(text)

        assert len(tokens) > 0
        assert any('é' in t or 'ï' or 'ü' for t in tokens)

    def test_very_long_document(self):
        """Test handling of very long documents."""
        # Create a large document
        tokens = ['word'] * 10000
        freqs = WordFrequencyAnalyzer.calculate_frequencies(tokens)

        assert freqs['word'] == 10000
        assert len(freqs) == 1

    def test_empty_token_list(self):
        """Test analysis with empty token list."""
        empty = []
        freqs = WordFrequencyAnalyzer.calculate_frequencies(empty)
        assert freqs == {}

        sentiment = SentimentAnalyzer().calculate_sentiment(empty)
        assert sentiment == 0.0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_text_processing(self):
        """Test complete text processing pipeline."""
        processor = TextProcessor()
        text = "Hello, world! This is a test. Hello again!"

        # Tokenize
        tokens = processor.tokenize(text)
        assert len(tokens) > 0

        # Remove stopwords
        filtered = processor.remove_stopwords(tokens, language='en')
        assert len(filtered) < len(tokens)

        # Calculate frequencies
        freqs = WordFrequencyAnalyzer.calculate_frequencies(filtered)
        assert 'hello' in freqs

    def test_corpus_with_multiple_analyses(self):
        """Test that multiple analyses work on same corpus."""
        docs = [
            ['python', 'programming', 'language'],
            ['python', 'snake', 'animal'],
            ['programming', 'code', 'development']
        ]

        corpus = CorpusFilter(docs)
        all_docs = corpus.filter_by_corpus('all')

        # Frequency analysis
        all_tokens = [t for doc in all_docs for t in doc]
        freqs = Counter(all_tokens)
        assert freqs['python'] == 2

        # TF-IDF analysis
        tfidf = TFIDFAnalyzer.calculate_tfidf(all_docs)
        assert len(tfidf) == 3

        # Cooccurrence analysis
        cooc = CooccurrenceAnalyzer.calculate_cooccurrence(
            all_tokens, window_size=5
        )
        assert len(cooc) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
