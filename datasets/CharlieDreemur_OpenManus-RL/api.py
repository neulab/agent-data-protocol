from typing import Any


def perform_action(action: str) -> dict:
    """Execute a text action in an interactive environment.

    Args:
    ----
        action: The environment action to perform, such as "go to desk 1".

    """
    pass


def get_search_movie(movie_name: Any) -> dict:
    """Search for a movie by name and return basic details."""
    pass


def get_movie_details(movie_id: Any) -> dict:
    """Get detailed information about a movie by ID."""
    pass


def get_movie_production_companies(movie_id: Any) -> dict:
    """Get the production companies of a movie by its ID."""
    pass


def get_movie_production_countries(movie_id: Any) -> dict:
    """Get the production countries of a movie by its ID."""
    pass


def get_movie_cast(movie_id: Any) -> dict:
    """Retrieve the top cast members from a movie by its ID."""
    pass


def get_movie_crew(movie_id: Any) -> dict:
    """Retrieve crew members from a movie by its ID."""
    pass


def get_movie_keywords(movie_id: Any) -> dict:
    """Get the keywords associated with a movie by ID."""
    pass


def get_search_person(person_name: Any) -> dict:
    """Search for a person by name."""
    pass


def get_person_details(person_id: Any) -> dict:
    """Get detailed information about a person by ID."""
    pass


def get_person_cast(person_id: Any) -> dict:
    """Retrieve movie cast roles for a person by their ID."""
    pass


def get_person_crew(person_id: Any) -> dict:
    """Retrieve movie crew roles for a person by their ID."""
    pass


def get_person_external_ids(person_id: Any) -> dict:
    """Get the external IDs for a person by ID."""
    pass


def get_movie_alternative_titles(movie_id: Any) -> dict:
    """Get alternative titles for a movie by ID."""
    pass


def get_movie_translation(movie_id: Any) -> dict:
    """Get description translations for a movie by ID."""
    pass


def check_valid_actions() -> dict:
    """Get supported actions for the current tool."""
    pass


def weather_get_120_hour_forecast_for_weather(
    lat: Any,
    lon: Any,
    lang: Any = None,
    hours: Any = None,
    units: Any = None,
) -> dict:
    """Return a weather forecast for up to 120 hours.

    Original tool name: weather.get_120_hour_forecast_for_weather.
    """
    pass


def pharmacies_de_garde_nc_health_for_pharmacies_de_garde_nc() -> dict:
    """Return the health status of the Pharmacies de garde NC application.

    Original tool name: pharmacies_de_garde_nc.health_for_pharmacies_de_garde_nc.
    """
    pass


def pharmacies_de_garde_nc_all_for_pharmacies_de_garde_nc() -> dict:
    """Return pharmacies de garde in Nouvelle-Calédonie.

    Original tool name: pharmacies_de_garde_nc.all_for_pharmacies_de_garde_nc.
    """
    pass


def app_store_new_free_ios_apps_for_app_store() -> dict:
    """Get a list of new free iOS apps.

    Original tool name: app_store.new_free_ios_apps_for_app_store.
    """
    pass
