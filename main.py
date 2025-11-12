import argparse
import yaml

from core.data_recorder import DataCollector, record_scenario

GLOBAL_CONFIG = "./config/global.yaml"
SENSORS_CONFIG = "./config/sensors.yaml"
SCENARIOS_CONFIG = "./config/scenarios.yaml"
V2E_CONFIG = "./config/v2e.yaml"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", "-s", type=int, default=1, help="The desired scenario's id (see scenarios.yaml for defined scenarios)")
    parser.add_argument("--all", action="store-true", help="Record all scenarios.")
    
    args = parser.parse_args()

    with open(GLOBAL_CONFIG, 'r') as global_cfg:
        global_config = yaml.safe_load(global_cfg)

    with open(SENSORS_CONFIG, 'r') as sensors_cfg:
        sensors_config = yaml.safe_load(sensors_cfg)

    with open(SCENARIOS_CONFIG, 'r') as scenarios_cfg:
        scenarios = yaml.safe_load(scenarios_cfg)

    with open(V2E_CONFIG, 'r', encoding='utf-8') as v2e_cfg:
        v2e_config = yaml.safe_load(v2e_cfg)


    if args.all:
        # Record all defined scenarios in ./config/scenarios.yaml
        collector = DataCollector(global_config, sensors_config, v2e_config, scenarios)
        collector.collect()
    else:
        # Record a single selected scenario
        record_scenario(args.scenario, global_config, sensors_config, v2e_config, scenarios)

