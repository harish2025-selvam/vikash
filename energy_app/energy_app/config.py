import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "energy_data.csv")

# Federated Learning Config
NUM_CLIENTS = 10
NUM_ROUNDS = 20
MU_FEDPROX = 0.01  # FedProx proximal term

# Required columns
REQUIRED_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

TARGET_COLUMN = "Global_active_power"

FEATURE_COLUMNS = [
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Model names
MODELS = ["Linear Regression", "Decision Tree", "Random Forest"]
BEST_MODEL_NAME = "Random Forest"

# Theme colors
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#7B2FBE",
    "accent": "#FF6B35",
    "bg": "#0A0E1A",
    "card": "#111827",
    "text": "#E2E8F0",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}
