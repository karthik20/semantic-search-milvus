#!/usr/bin/env python
"""
Generate sample data for the help_support collection.
"""

import json
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Categories and related tags for help topics
CATEGORIES = [
    {
        "name": "Account Management",
        "topics": [
            "Opening Accounts", "Closing Accounts", "Account Types", "Statements",
            "Fees", "Minimum Balance", "Joint Accounts", "Account Verification"
        ],
        "tags": ["accounts", "banking"]
    },
    {
        "name": "Online & Mobile Banking",
        "topics": [
            "App Installation", "Login Issues", "Password Reset", "Transaction History",
            "Mobile Check Deposit", "Push Notifications", "Biometric Authentication"
        ],
        "tags": ["digital", "online banking", "mobile app"]
    },
    {
        "name": "Cards",
        "topics": [
            "Activation", "PIN Management", "Lost Cards", "Card Replacement", 
            "Contactless Payments", "Credit Limits", "Card Rewards", "Foreign Transactions"
        ],
        "tags": ["cards", "debit cards", "credit cards"]
    },
    {
        "name": "Loans & Mortgages",
        "topics": [
            "Loan Application", "Repayment Options", "Refinancing", "Interest Rates",
            "Early Repayment", "Mortgage Calculator", "Home Equity Loans"
        ],
        "tags": ["loans", "mortgages", "financing"]
    },
    {
        "name": "Payments & Transfers",
        "topics": [
            "Bill Pay", "Wire Transfers", "Recurring Payments", "Transfer Limits",
            "Payment Scheduling", "International Transfers", "Transfer Cancellation"
        ],
        "tags": ["payments", "transfers", "bill pay"]
    },
    {
        "name": "Security",
        "topics": [
            "Fraud Prevention", "Identity Theft", "Secure Messaging", "Transaction Alerts",
            "Account Monitoring", "Security Questions", "Device Management"
        ],
        "tags": ["security", "fraud", "protection"]
    },
    {
        "name": "Investments",
        "topics": [
            "Investment Accounts", "Portfolio Management", "Retirement Planning", 
            "Stocks & Bonds", "Mutual Funds", "Investment Advisory", "Market Updates"
        ],
        "tags": ["investments", "wealth", "retirement"]
    }
]

def generate_help_content():
    """Generate realistic help content based on banking topics."""
    # Select a random category
    category = random.choice(CATEGORIES)
    
    # Select a random topic from the category
    topic = random.choice(category["topics"])
    
    # Generate a title based on the topic
    if random.random() < 0.3:
        title = f"How to {topic.lower()}"
    elif random.random() < 0.6:
        title = f"Understanding your {topic.lower()}"
    else:
        title = f"{topic}: Frequently asked questions"
    
    # Generate a URL based on the category and topic
    url_path = f"{category['name'].lower().replace(' & ', '-').replace(' ', '-')}/{topic.lower().replace(' ', '-')}"
    url = f"https://bank.example.com/help/{url_path}"
    
    # Generate ID
    doc_id = f"help-{fake.unique.random_int(min=100, max=999)}"
    
    # Generate content with multiple paragraphs
    paragraphs = []
    for _ in range(random.randint(2, 4)):
        paragraphs.append(fake.paragraph(nb_sentences=random.randint(3, 8)))
    content = "\n\n".join(paragraphs)
    
    # Select tags
    tags = category["tags"].copy()
    if random.random() < 0.3:
        tags.append(random.choice(["faq", "quick help", "getting started", "support"]))
    
    return {
        "id": doc_id,
        "url": url,
        "title": title,
        "content": content,
        "tags": tags
    }

def generate_help_support_data(count=100):
    """Generate a specified number of help support documents."""
    return [generate_help_content() for _ in range(count)]

if __name__ == "__main__":
    # Generate 100 documents
    data = generate_help_support_data(100)
    
    # Add the initial samples at the beginning for diversity
    with open('data/help_support_sample.json', 'r') as f:
        initial_samples = json.load(f)
    
    # Combine samples
    all_data = initial_samples + data
    
    # Write to file
    with open('data/help_support_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Generated {len(all_data)} help support documents in data/help_support_data.json")
