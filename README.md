# CARLA Simulation Data Collection Tool

This project provides a configurable tool for running and recording simulations in CARLA, using YAML-based configuration files to define simulation parameters, scenarios, and sensors.

--------------------------------------------------------------------------------
## 🚀 Overview

This tool allows you to:

- Define simulation global settings (client, traffic manager, simulation mode, etc.)
- Configure multiple scenarios (maps, weather, number of actors, etc.)
- Define custom sensor setups
- Enable V2E (Vision-to-Event) mode for advanced event-based camera simulation
- Run and record individual or multiple scenarios automatically

--------------------------------------------------------------------------------
## ⚙️ Setup (Windows)

### 1. Install CARLA

Follow the official CARLA installation guide for Windows:
https://carla.readthedocs.io/en/0.9.15/start_quickstart/

Note:
This tool was developed and tested using CARLA 0.9.15.
Compatibility with other versions is not guaranteed.

### 2. Configure the Simulation

Before running, make sure the following YAML files are properly configured:

- global.yaml — defines global simulation behavior (host, ports, save directory, etc.)
- scenarios.yaml — defines which scenarios to run, including maps and weather
- sensors.yaml — defines sensors and their callbacks
- v2e.yaml — defines Vision-to-Event (V2E) camera settings (if enabled)

### 3. Start the CARLA Server

In one terminal or PowerShell window, navigate to your CARLA installation folder and start the server:

```./CarlaUE4.exe```

Keep the carla window open — the CARLA server must be running while you use the tool.

--------------------------------------------------------------------------------
## ▶️ Running the Simulation

Once the CARLA server is running, open a new terminal or PowerShell window and navigate to the project directory.

### Run a Single Scenario

To run and record a single scenario (for example, Scenario 1):

```python main.py -s 1```

Here, 1 corresponds to the scenario defined in scenarios.yaml
(under the field "Scenario 1").

### Run All Scenarios

To automatically record all defined scenarios sequentially:

```python main.py --all```

--------------------------------------------------------------------------------
## 📁 Output

All collected data will be saved in the directory defined in global.yaml:
```
collector:
  save_dir: "./SAVED_DATA"
```

Each scenario’s recordings will be stored in its own subfolder within this directory.

--------------------------------------------------------------------------------
## 🧠 Notes

- Make sure the CARLA server window is fully loaded before starting the simulation.
- Synchronous mode (configured in global.yaml) is recommended for consistent results.
- When using V2E, ensure that:
  - v2e_enabled is set to True in global.yaml
  - An RGB sensor is defined in sensors.yaml with the proper callback

--------------------------------------------------------------------------------
## 🧾 CARLA License

This tool is intended for research and educational use with the CARLA Simulator.
Refer to CARLA’s license terms for usage restrictions:
https://carla.readthedocs.io/en/latest/about/

--------------------------------------------------------------------------------
