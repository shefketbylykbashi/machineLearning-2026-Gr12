"""
Main Entry Point - Clean Architecture ETL Pipeline
Refactored from original monolithic script to follow clean architecture principles.
Calls the modular ETL pipeline while maintaining backward compatibility.
"""
from .pipeline import main

if __name__ == "__main__":
    main()
