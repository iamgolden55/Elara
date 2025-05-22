import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from config import settings

async def check_appointment_availability(doctor_id: str, date: str) -> Dict[str, Any]:
    """
    Check appointment availability for a specific doctor and date
    
    Args:
        doctor_id: ID of the doctor to check
        date: Date string in format YYYY-MM-DD
        
    Returns:
        Dictionary with available time slots
    """
    api_key = settings.BOOKING_API_KEY
    
    if not api_key:
        return {
            "error": "API key not configured",
            "message": "Booking API key is not configured in settings"
        }
    
    # Example API URL (replace with actual booking system API)
    api_url = f"https://api.medicalbooking.example/v1/availability"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                api_url,
                params={
                    "doctor_id": doctor_id,
                    "date": date
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

async def book_appointment(doctor_id: str, patient_id: str, datetime_slot: str) -> Dict[str, Any]:
    """
    Book an appointment for a patient with a doctor
    
    Args:
        doctor_id: ID of the doctor
        patient_id: ID of the patient
        datetime_slot: Datetime slot in format YYYY-MM-DDTHH:MM:SS
        
    Returns:
        Dictionary with booking confirmation
    """
    api_key = settings.BOOKING_API_KEY
    
    if not api_key:
        return {
            "error": "API key not configured",
            "message": "Booking API key is not configured in settings"
        }
    
    # Example API URL (replace with actual booking system API)
    api_url = f"https://api.medicalbooking.example/v1/bookings"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                api_url,
                json={
                    "doctor_id": doctor_id,
                    "patient_id": patient_id,
                    "datetime": datetime_slot
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            # Handle response
            if response.status_code == 201:
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

async def get_nearby_doctors(specialty: str, location: str, radius: int = 10) -> Dict[str, Any]:
    """
    Find nearby doctors based on specialty and location
    
    Args:
        specialty: Medical specialty (e.g., 'cardiology', 'pediatrics')
        location: Location string (e.g., city or zip code)
        radius: Search radius in miles/km
        
    Returns:
        Dictionary with list of doctors
    """
    api_key = settings.BOOKING_API_KEY
    
    if not api_key:
        return {
            "error": "API key not configured",
            "message": "Booking API key is not configured in settings"
        }
    
    # Example API URL (replace with actual doctor search API)
    api_url = f"https://api.medicalbooking.example/v1/doctors/search"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                api_url,
                params={
                    "specialty": specialty,
                    "location": location,
                    "radius": radius
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
