from livekit.agents import function_tool

@function_tool(name="gethotel_details")
async def gethotel_details(destination_city: str) -> str:
    """Get hotel details and recommendations for a given destination city. Use this when the user asks for accommodation options."""
    return f"Hotel Options in {destination_city}: 1) The Grand Plaza (5-star), 2) Oceanview Villa (luxury), 3) City Center Inn (4-star)."

@function_tool(name="best_TravelOption")
async def best_TravelOption(source: str, destination: str) -> str:
    """Get the best travel options including flights, trains, and buses between a source and destination."""
    return f"Travel Options from {source} to {destination}: Flight (AirExpress, 2h, $150), Train (ScenicRail, 6h, $60), Bus (CityTransit, 10h, $30)."

@function_tool(name="locationbasedactivitydetail")
async def locationbasedactivitydetail(location: str) -> str:
    """Get popular location-based activities and recommendations for a specific destination."""
    return f"Activities in {location}: Adventure (Mountain Zip-lining), Sightseeing (Historic Downtown Walk), Relaxing (Sunset Cruise & Spa)."




