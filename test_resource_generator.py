import asyncio
import sys
import os

# Add the current directory to sys.path to allow importing services
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from services.resource_generator_service import generate_resources_for_skill
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

async def test_generation():
    skill = "Flutter"
    print(f"Testing resource generation for: {skill}...")
    try:
        resources = await generate_resources_for_skill(skill)
        print("\nGenerated Resources:")
        for i, res in enumerate(resources, 1):
            print(f"{i}. {res['title']} ({res['platform']})")
            print(f"   URL: {res['url']}")
            print(f"   Desc: {res['description']}")
            print("-" * 30)
            
        # Verify fallback logic
        print("\nTesting fallback for empty skill name...")
        fallback_resources = await generate_resources_for_skill("")
        print(f"Fallback URL: {fallback_resources[0]['url']}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_generation())
