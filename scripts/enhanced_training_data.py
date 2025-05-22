#!/usr/bin/env python3
"""
🧠 Elara AI - Enhanced Training Data Generator 🧠
Generates high-quality training data for medical LoRA fine-tuning
with improved reliability and fact checking.

This script addresses the following improvements:
1. Increased training data quantity (5000+ examples)
2. Better data quality with fact verification
3. Proper validation splits (train/val/test)
4. RAG-compatible formatting
5. Confidence scoring preparation
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the project root directory
PROJECT_ROOT = Path(__file__).parents[1].absolute()
DATA_DIR = PROJECT_ROOT / "data"
HIGH_QUALITY_DIR = DATA_DIR / "high_quality"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
RAG_DIR = DATA_DIR / "rag"

# Ensure directories exist
for directory in [DATA_DIR, HIGH_QUALITY_DIR, PROCESSED_DIR, TRAINING_DATA_DIR, RAG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class EnhancedTrainingDataGenerator:
    """Generate high-quality, reliable training data for medical AI fine-tuning"""
    
    def __init__(self, 
                 num_examples: int = 5000,
                 user_types: List[str] = ["patient", "medical_professional"],
                 validation_split: float = 0.1,
                 test_split: float = 0.1,
                 quality_threshold: float = 0.7,
                 max_input_length: int = 512,
                 max_output_length: int = 1024,
                 seed: int = 42):
        """
        Initialize the training data generator with improved parameters
        
        Args:
            num_examples: Total number of examples to generate
            user_types: List of user types to generate data for
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            quality_threshold: Minimum quality score for articles (0-1)
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            seed: Random seed for reproducibility
        """
        self.num_examples = num_examples
        self.user_types = user_types
        self.validation_split = validation_split
        self.test_split = test_split
        self.quality_threshold = quality_threshold
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Initialize containers
        self.medical_articles = []
        self.medical_qa_pairs = []
        self.medical_topics = defaultdict(list)
        self.medical_terms = set()
        
        # Statistics tracking
        self.stats = {
            "articles_loaded": 0,
            "articles_filtered": 0,
            "qa_pairs_generated": 0,
            "topics_covered": 0,
            "avg_quality_score": 0.0,
            "train_examples": 0,
            "val_examples": 0,
            "test_examples": 0
        }
        
        logger.info(f"Initialized Enhanced Training Data Generator targeting {num_examples} examples")
    
    def load_medical_articles(self) -> None:
        """
        Load and filter high-quality medical articles from the data directory
        with improved reliability checks
        """
        logger.info("Loading medical articles from data directory...")
        
        # Load articles from high_quality directory
        article_files = list(HIGH_QUALITY_DIR.glob("*.json"))
        if not article_files:
            logger.warning(f"No article files found in {HIGH_QUALITY_DIR}")
            raise FileNotFoundError(f"No medical articles found in {HIGH_QUALITY_DIR}")
        
        total_articles = 0
        quality_articles = 0
        quality_scores = []
        
        # Process each file
        for file_path in tqdm(article_files, desc="Loading article files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                
                # Ensure articles is a list
                if isinstance(articles, dict):
                    articles = [articles]
                
                for article in articles:
                    total_articles += 1
                    
                    # Check article quality using enhanced criteria
                    quality_score = self._calculate_quality_score(article)
                    quality_scores.append(quality_score)
                    
                    if quality_score >= self.quality_threshold:
                        # Add article topic for better organization
                        topic = self._extract_topic(article)
                        if topic:
                            article["topic"] = topic
                            self.medical_topics[topic].append(article)
                        
                        # Extract medical terms for verification
                        terms = self._extract_medical_terms(article)
                        self.medical_terms.update(terms)
                        
                        # Add the high-quality article
                        self.medical_articles.append(article)
                        quality_articles += 1
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Calculate statistics
        self.stats["articles_loaded"] = total_articles
        self.stats["articles_filtered"] = quality_articles
        self.stats["avg_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        self.stats["topics_covered"] = len(self.medical_topics)
        
        logger.info(f"Loaded {quality_articles} high-quality articles from {total_articles} total articles")
        logger.info(f"Average quality score: {self.stats['avg_quality_score']:.3f}")
        logger.info(f"Covering {len(self.medical_topics)} distinct medical topics")
    
    def _calculate_quality_score(self, article: Dict[str, Any]) -> float:
        """
        Calculate enhanced quality score for medical articles
        using multiple factors
        
        Args:
            article: Medical article dictionary
            
        Returns:
            quality_score: Float between 0-1 indicating quality
        """
        score = 0.0
        
        # Abstract presence and quality (40% weight)
        abstract = article.get('abstract', '')
        if abstract:
            # Base points for having an abstract
            score += 0.2
            
            # Extra points for longer, more detailed abstracts
            if len(abstract) > 500:
                score += 0.1
            if len(abstract) > 1000:
                score += 0.1
        
        # Title quality (15% weight)
        title = article.get('title', '')
        if title:
            if len(title) > 10:
                score += 0.05
            if len(title) > 20:
                score += 0.05
            if len(title) > 30:
                score += 0.05
        
        # References/citations (15% weight)
        refs = article.get('references', [])
        if isinstance(refs, list) and len(refs) > 0:
            score += min(0.15, len(refs) * 0.01)  # Up to 0.15 for 15+ references
        
        # Publication metadata (10% weight)
        if article.get('journal') or article.get('publisher'):
            score += 0.05
        if article.get('publication_date') or article.get('year'):
            score += 0.05
        
        # Medical keywords presence (20% weight)
        medical_keywords = [
            'patient', 'treatment', 'diagnosis', 'therapy', 'clinical',
            'disease', 'symptoms', 'syndrome', 'prognosis', 'medication',
            'virus', 'bacteria', 'infection', 'inflammation', 'chronic',
            'acute', 'cancer', 'surgery', 'medicine', 'healthcare'
        ]
        
        text = (title + " " + abstract).lower()
        keyword_count = sum(1 for keyword in medical_keywords if keyword in text)
        score += (keyword_count / len(medical_keywords)) * 0.2
        
        return min(1.0, score)  # Cap at 1.0
    
    def _extract_topic(self, article: Dict[str, Any]) -> str:
        """
        Extract the primary medical topic from an article
        
        Args:
            article: Medical article dictionary
            
        Returns:
            topic: Primary medical topic or specialty
        """
        # List of common medical specialties and topics
        specialties = [
            "cardiology", "neurology", "oncology", "pediatrics", 
            "psychiatry", "dermatology", "endocrinology", "gastroenterology",
            "gynecology", "hematology", "immunology", "infectious disease",
            "nephrology", "pulmonology", "rheumatology", "surgery",
            "urology", "orthopedics", "emergency medicine", "family medicine"
        ]
        
        # Try to extract from keywords or MeSH terms first
        keywords = article.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
            
        for keyword in keywords:
            keyword = keyword.lower()
            for specialty in specialties:
                if specialty in keyword:
                    return specialty
        
        # Try title and abstract
        text = (article.get('title', '') + " " + article.get('abstract', '')).lower()
        
        # Count occurrences of each specialty
        topic_counts = {}
        for specialty in specialties:
            count = text.count(specialty)
            if count > 0:
                topic_counts[specialty] = count
        
        # Return most frequent specialty or "general medicine" if none found
        if topic_counts:
            return max(topic_counts.items(), key=lambda x: x[1])[0]
        
        return "general medicine"
    
    def _extract_medical_terms(self, article: Dict[str, Any]) -> List[str]:
        """
        Extract key medical terms from an article for verification
        
        Args:
            article: Medical article dictionary
            
        Returns:
            terms: List of medical terms
        """
        terms = []
        text = (article.get('title', '') + " " + article.get('abstract', '')).lower()
        
        # Extract common medical term patterns
        # This is a simplified approach - a more robust NLP approach would be better
        # in a production system
        
        # Look for terms in brackets
        import re
        bracket_terms = re.findall(r'\(([^)]+)\)', text)
        for term in bracket_terms:
            if len(term) > 3 and not term.isdigit():
                terms.append(term.strip())
        
        # Look for capitalized multi-word terms
        cap_terms = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', 
                              article.get('abstract', ''))
        terms.extend([term.lower() for term in cap_terms])
        
        # Add keywords if available
        keywords = article.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        terms.extend([k.lower() for k in keywords if isinstance(k, str)])
        
        return list(set(terms))  # Remove duplicates
    
    def generate_training_data(self) -> None:
        """
        Generate enhanced training data from loaded medical articles
        with improved quality and reliability
        """
        if not self.medical_articles:
            self.load_medical_articles()
        
        logger.info(f"Generating training data for {len(self.user_types)} user types...")
        
        training_data = []
        
        # Use topic-balanced approach
        examples_per_topic = max(1, self.num_examples // max(1, len(self.medical_topics)))
        
        # For each topic, generate examples for each user type
        for topic, articles in tqdm(self.medical_topics.items(), desc="Processing topics"):
            # Generate examples for each user type
            for user_type in self.user_types:
                # Generate multiple QA pairs for this topic and user type
                topic_examples = self._generate_topic_examples(
                    topic, 
                    articles, 
                    examples_per_topic // len(self.user_types),
                    user_type
                )
                
                # Add to training data
                training_data.extend(topic_examples)
        
        # If we have fewer examples than requested, generate more from generic templates
        if len(training_data) < self.num_examples:
            logger.info(f"Generated {len(training_data)} examples from articles. Adding synthetic examples to reach target...")
            
            additional_needed = self.num_examples - len(training_data)
            synthetic_examples = self._generate_synthetic_examples(additional_needed)
            training_data.extend(synthetic_examples)
        
        # Shuffle and split data
        random.shuffle(training_data)
        
        # Split into train, validation, and test sets
        val_size = int(len(training_data) * self.validation_split)
        test_size = int(len(training_data) * self.test_split)
        train_size = len(training_data) - val_size - test_size
        
        train_data = training_data[:train_size]
        val_data = training_data[train_size:train_size + val_size]
        test_data = training_data[train_size + val_size:]
        
        # Update statistics
        self.stats["qa_pairs_generated"] = len(training_data)
        self.stats["train_examples"] = len(train_data)
        self.stats["val_examples"] = len(val_data)
        self.stats["test_examples"] = len(test_data)
        
        logger.info(f"Generated {len(training_data)} total examples")
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
        
        # Save data
        self._save_training_data(train_data, val_data, test_data)
        
        # Save statistics
        self._save_statistics()
    
    def _generate_topic_examples(self, 
                               topic: str, 
                               articles: List[Dict[str, Any]], 
                               num_examples: int,
                               user_type: str) -> List[Dict[str, Any]]:
        """
        Generate QA examples for a specific medical topic
        
        Args:
            topic: Medical topic
            articles: List of articles on this topic
            num_examples: Number of examples to generate
            user_type: User type (patient or medical_professional)
            
        Returns:
            examples: List of QA examples
        """
        examples = []
        
        # Generate question templates based on topic
        question_templates = self._get_question_templates(topic, user_type)
        

