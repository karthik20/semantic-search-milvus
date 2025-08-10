#!/usr/bin/env python
"""
Generate sample data for the services collection.
"""

import json
import random
import string

# Service categories and related information
SERVICES = [
    {
        "category": "Accounts",
        "types": ["Savings", "Checking", "Money Market", "Certificate of Deposit", "High-Yield Savings", 
                 "Student Account", "Senior Account", "Business Checking", "Business Savings"],
        "intent_prefix": "account_opening",
        "url_prefix": "accounts",
        "descriptions": [
            "Earn {interest}% APY on balances with our {name}. {features}. {benefits}.",
            "Our {name} offers {features} with no minimum balance requirement. {benefits}.",
            "The {name} is designed for {audience}. {features}. {benefits}.",
            "{name} - our most popular account option with {features}. {benefits}."
        ]
    },
    {
        "category": "Cards",
        "types": ["Credit Card", "Rewards Card", "Cash Back Card", "Travel Card", "Student Card", 
                 "Business Credit Card", "Secured Credit Card", "Debit Card", "Prepaid Card"],
        "intent_prefix": "card_application",
        "url_prefix": "cards",
        "descriptions": [
            "Our {name} offers {reward_rate}% {reward_type} on all purchases. {features}. {benefits}.",
            "The {name} has no annual fee and {features}. Perfect for {audience}. {benefits}.",
            "Earn {reward_rate}X points on {category_spend} with the {name}. {features}. {benefits}.",
            "{name} - designed with {audience} in mind, offering {features}. {benefits}."
        ]
    },
    {
        "category": "Loans",
        "types": ["Personal Loan", "Auto Loan", "Student Loan", "Home Equity Loan", "Mortgage", 
                 "Business Loan", "Line of Credit", "Debt Consolidation Loan"],
        "intent_prefix": "loan_application",
        "url_prefix": "loans",
        "descriptions": [
            "Our {name} offers competitive rates starting at {rate}% APR. {features}. {benefits}.",
            "Get approved for a {name} with flexible terms from {min_term} to {max_term} years. {features}. {benefits}.",
            "The {name} provides funds from ${min_amount} to ${max_amount}. {features}. {benefits}.",
            "{name} - fast approval and {features}. {benefits}."
        ]
    },
    {
        "category": "Investments",
        "types": ["Retirement Account", "IRA", "401(k) Rollover", "Brokerage Account", "Managed Portfolio", 
                 "Robo-Advisor", "College Savings Plan", "Mutual Funds"],
        "intent_prefix": "investment_opening",
        "url_prefix": "investments",
        "descriptions": [
            "Our {name} helps you build wealth with {features}. {benefits}.",
            "The {name} offers {features} with professional management. {benefits}.",
            "Start investing with as little as ${min_amount} with our {name}. {features}. {benefits}.",
            "{name} - diversify your portfolio with {features}. {benefits}."
        ]
    },
    {
        "category": "Insurance",
        "types": ["Life Insurance", "Home Insurance", "Auto Insurance", "Health Insurance", 
                 "Travel Insurance", "Pet Insurance", "Business Insurance"],
        "intent_prefix": "insurance_quote",
        "url_prefix": "insurance",
        "descriptions": [
            "Protect what matters with our {name}. {features}. {benefits}.",
            "Our {name} offers comprehensive coverage with {features}. {benefits}.",
            "Get peace of mind with {name}, providing {features}. {benefits}.",
            "{name} - affordable premiums and {features}. {benefits}."
        ]
    }
]

# Features, benefits, and audiences to mix into descriptions
FEATURES = [
    "24/7 online access", "mobile banking support", "monthly statements", "automatic transfers",
    "free bill pay", "zero liability protection", "fraud monitoring", "customizable alerts",
    "overdraft protection", "ATM fee reimbursements", "no foreign transaction fees", "instant notifications",
    "account insights", "financial planning tools", "budgeting assistance", "round-up savings",
    "direct deposit", "free money transfers", "no monthly maintenance fees", "high-yield interest rates"
]

BENEFITS = [
    "Save time and money", "Achieve your financial goals faster", "Enjoy peace of mind",
    "Take control of your finances", "Bank on your terms", "Access your money anywhere, anytime",
    "Grow your wealth effortlessly", "Stay protected against fraud", "Perfect for busy lifestyles",
    "Designed to simplify your banking experience", "Backed by our satisfaction guarantee",
    "Supported by award-winning customer service", "Trusted by millions of customers",
    "Rated #1 in customer satisfaction"
]

AUDIENCES = [
    "students", "young professionals", "families", "business owners", "seniors", 
    "travelers", "first-time homebuyers", "investors", "savers", "high-net-worth individuals"
]

def random_id(prefix="svc"):
    """Generate a random service ID."""
    digits = ''.join(random.choice(string.digits) for _ in range(3))
    return f"{prefix}-{digits}"

def generate_service():
    """Generate a random banking service."""
    # Select a random service category
    service_category = random.choice(SERVICES)
    
    # Select a random type from the category
    service_type = random.choice(service_category["types"])
    
    # Create a service name
    name_prefixes = ["Premier", "Essential", "Elite", "Standard", "Select", "Advantage", 
                     "Priority", "Preferred", "Signature", "Ultimate", "Basic", "Premium"]
    
    name = f"{random.choice(name_prefixes)} {service_type}"
    
    # Create service ID
    service_id = random_id()
    
    # Create URL
    type_url = service_type.lower().replace(" ", "-")
    url = f"https://bank.example.com/{service_category['url_prefix']}/{type_url}"
    
    # Create intent_entity
    intent_entity = f"{service_category['intent_prefix']}-{service_type.lower().replace(' ', '_')}"
    
    # Create description
    description_template = random.choice(service_category["descriptions"])
    
    description = description_template.format(
        name=name,
        interest=round(random.uniform(0.1, 3.5), 2),
        rate=round(random.uniform(3.5, 18.9), 2),
        reward_rate=round(random.uniform(1, 5), 1),
        reward_type=random.choice(["cash back", "points", "miles", "rewards"]),
        category_spend=random.choice(["dining", "travel", "groceries", "gas", "online purchases"]),
        min_amount=random.choice([100, 250, 500, 1000, 2500]),
        max_amount=random.choice([10000, 25000, 50000, 100000]),
        min_term=random.choice([1, 2, 3, 5]),
        max_term=random.choice([10, 15, 20, 30]),
        features=", ".join(random.sample(FEATURES, 2)),
        benefits=random.choice(BENEFITS),
        audience=random.choice(AUDIENCES)
    )
    
    return {
        "service_id": service_id,
        "url": url,
        "name": name,
        "description": description,
        "intent_entity": intent_entity
    }

def generate_services_data(count=100):
    """Generate a specified number of banking services."""
    return [generate_service() for _ in range(count)]

if __name__ == "__main__":
    # Generate 100 services
    data = generate_services_data(100)
    
    # Add the initial samples at the beginning for diversity
    with open('data/services_sample.json', 'r') as f:
        initial_samples = json.load(f)
    
    # Combine samples
    all_data = initial_samples + data
    
    # Write to file
    with open('data/services_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Generated {len(all_data)} services in data/services_data.json")
