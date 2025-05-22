import httpx
from typing import Dict, List, Any, Optional
from config import settings

async def search_drug_info(drug_name: str) -> Dict[str, Any]:
    """
    Search for drug information using an external API
    
    Args:
        drug_name: Name of the drug to search for
        
    Returns:
        Dictionary with drug information
    """
    api_key = settings.DRUG_API_KEY
    
    if not api_key:
        return {
            "error": "API key not configured",
            "message": "Drug API key is not configured in settings"
        }
    
    # Example API URL (replace with actual drug database API)
    api_url = f"https://api.drugdatabase.example/v1/drugs"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                api_url,
                params={
                    "name": drug_name,
                    "limit": 5
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            # Handle response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                }
                
        except Exception as e:
            return {
                "error": "Connection error",
                "message": str(e)
            }

async def get_drug_interactions(drug_list: List[str]) -> Dict[str, Any]:
    """
    Check for interactions between multiple drugs
    
    Args:
        drug_list: List of drug names to check for interactions
        
    Returns:
        Dictionary with interaction information
    """
    api_key = settings.DRUG_API_KEY
    
    if not api_key:
        return {
            "error": "API key not configured",
            "message": "Drug API key is not configured in settings"
        }
    
    # Example API URL (replace with actual drug interaction API)
    api_url = f"https://api.drugdatabase.example/v1/interactions"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                api_url,
                json={
                    "drugs": drug_list
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            # Handle response
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "message": response.text
                }
                
        except Exception as e:
            return {
                "error": "Connection error",
                "message": str(e)
            }
