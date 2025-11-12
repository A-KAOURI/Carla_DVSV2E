import glob
import os
import sys

import carla.libcarla

try:
    sys.path.append(glob.glob('/home/abdou/CARLA_0.9.15/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging

import random
import time
from pathlib import Path

import queue
from queue import Empty

import imageio.v2 as imageio
import numpy as np

from flow_vis import flow_to_color, flow_uv_to_colors
import cv2
from matplotlib import cm
from matplotlib import colors

from functools import partial
from tqdm import tqdm
import weakref

import callbacks
from v2e.emulator import EventEmulator
from visualization import events_to_event_image



# For measuring performance
# time_per_event_list = []

def visualize_optical_flow(flow, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    # flow = flow.transpose(1,2,0)
    flow[np.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = np.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = np.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=np.pi*2
    hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)

    return rgb, (mag.min(), mag.max())


# @todo cannot import these directly.
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class DataCollector():
    # The data collector must connect with CARLA, retreive the world and pass everything to each DataRecorder
    # A separate data recorder is used for each Scenario to record a sequence
    def __init__(self, global_config, sensors_config, v2e_config, scenarios):
        # config
        self.config = global_config
        self.sensors_config = sensors_config
        self.scenarios = scenarios
        self.v2e_config = v2e_config

        # save_dir
        self.save_dir = Path(self.config["collector"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # client
        # self.client =  carla.Client(self.config["client"]["host"], self.config["client"]["port"])
        # self.client.set_timeout(self.config["client"]["timeout"])


    def collect(self):
        # Get maps
        for scenario_args in self.scenarios.values():
            recorder = DataRecorder(scenario_args, self.config, self.sensors_config, self.v2e_config)
            recorder.record()

def record_scenario(scenario_id:int, global_config, sensors_config, v2e_config, scenarios):
    scenario_name = f"Scenario {scenario_id}"
    assert scenario_name in scenarios

    recorder = DataRecorder(scenarios[scenario_name], global_config, sensors_config, v2e_config)
    recorder.record()

class DataRecorder():
    def __init__(self, args, global_config, sensors_config, v2e_config):     
        self.args = args
        self.global_config = global_config
        self.sensors_config = sensors_config

        save_dir_root = global_config["collector"]["save_dir"]
        self.save_dir = Path(save_dir_root) / args["name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.simulated_seconds = args["simulated_seconds"]

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        # Carla Client
        host = global_config["client"]["host"]
        port = global_config["client"]["port"]
        self.client = carla.Client(host, port)
        self.client.set_timeout(global_config["client"]["timeout"])
        
        # World and Traffic manager
        self.world = self.client.load_world(args['map'])
        self.spectator = self.world.get_spectator()

        self.traffic_manager = self.client.get_trafficmanager(global_config["traffic_manager"]["tm_port"])
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if global_config["traffic_manager"]["respawn"]:
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        if global_config["traffic_manager"]["hybrid"]:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
        
        rand_seed = self.global_config["collector"]["random_seed"]
        if rand_seed is not None:
            self.traffic_manager.set_random_device_seed(rand_seed)
        
        random.seed(rand_seed if rand_seed is not None else int(time.time()))
        self.synchronous_master = False

        # Synchrony and V2E settings
        self.synchronous = global_config["simulation"]["synchronous"]
        self.v2e_enabled = global_config["simulation"]["v2e_enabled"]
        self.v2e_multiplier = int(global_config["simulation"]["v2e_multiplier"])
        self.timestep = global_config["simulation"]["timestep"]

        settings = self.world.get_settings()
        if self.synchronous:
            self.synchronous_master = True
            self.traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True

            if self.v2e_enabled:
                self.hfps_timestep = self.timestep / self.v2e_multiplier

                settings.fixed_delta_seconds = self.hfps_timestep
                self.total_frames = int(self.simulated_seconds // self.hfps_timestep)

            else:
                settings.fixed_delta_seconds = self.timestep
                self.total_frames = int(self.simulated_seconds // self.timestep)

        else:
            raise NotImplementedError

        # Rendering
        if global_config["simulation"]["no_rendering"]:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        # Weather settings
        if args["weather_preset"] is not None:
            assert hasattr(carla.WeatherParameters, args["weather_preset"])
            weather = getattr(carla.WeatherParameters, args["weather_preset"])
            
            if args["weather_settings"] is not None:
                for key, value in args["weather_settings"].items():
                    setattr(weather, key, float(value))
        
        else:
            weather = carla.WeatherParameters(**args["weather_settings"])            

        self.world.set_weather(weather)

        # Blueprints
        self.blueprints = get_actor_blueprints(self.world, global_config["simulation"]["filterv"], global_config["simulation"]["generationv"])
        self.blueprintsWalkers = get_actor_blueprints(self.world, global_config["simulation"]["filterw"], global_config["simulation"]["generationw"])
        
        if global_config["traffic_manager"]["safe"]:
            self.blueprints = [x for x in self.blueprints if x.get_attribute('base_type') == 'car']

        self.blueprints = sorted(self.blueprints, key=lambda bp: bp.id)

        # Spawn Points
        self.spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.spawn_points)

        if args["number_of_vehicles"] < number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif args["number_of_vehicles"] > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args["number_of_vehicles"], number_of_spawn_points)
            args["number_of_vehicles"] = number_of_spawn_points

        # Spawn main vehicle
        hero_bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        hero_transform = random.choice(self.spawn_points)
        self.hero = self.world.spawn_actor(hero_bp, hero_transform)
        self.hero.set_autopilot(True)
        self.traffic_manager.auto_lane_change(self.hero, False)

        # Spawn NPC vehicles and pedestrians
        self.spawn_vehicles()
        self.spawn_walkers()

        # Spawn sensors
        self.sensors = {}
        self.sensors_bp = {}
        self.data_save_dirs = {}
        self.visual_dirs = {}
        self.sensor_queues = {}
        self.spawn_sensors()


        # Traffic lights
        if global_config["simulation"]["traffic_lights_off"]:
            # Get all traffic light actors
            traffic_lights = self.world.get_actors().filter("traffic.traffic_light")

            # Turn them all off
            for light in traffic_lights:
                light.set_state(carla.TrafficLightState.Off)
                light.freeze(True)  # freeze so it doesn’t cycle back


        # V2E Simulator
        self.v2e = EventEmulator(**v2e_config)

        self.bgr_buffer = queue.Queue()
        self.current_events = np.zeros((0, 4), dtype=np.float32)

        self.v2e_savedir = self.save_dir / "Events (v2e)"
        self.event_visdir = self.save_dir / "visualizations/V2E"
        # self.rgb_savedir = self.save_dir / "RGB_no_blurr"

        if self.v2e_enabled:
            self.v2e_savedir.mkdir(parents=True, exist_ok=True)
                
        if self.global_config["simulation"]["v2e_visualize"]:
            self.event_visdir.mkdir(parents=True, exist_ok=True)

        # Simulation tracking variables
        self.first_frame = None

        self.bgr_count = 0
        self.start_time = None

        self.dvs_prev_ts = None
        self.dvs_curr_ts = None
        

    def spawn_sensors(self):
        sensors_location = carla.Location(**self.global_config["sensors"]["location"])
        img_height = self.global_config["sensors"]["resolution"]["height"]
        img_width = self.global_config["sensors"]["resolution"]["width"]

        for s_name, conf in self.sensors_config.items():
            if not conf['enable']:
                # Skip disabled sensors
                continue

            if not self.v2e_enabled and s_name == "RGB_hfps":
                # Skip the high fps RGB sensor if it is not used
                continue
            
            # Get blueprint and setup resolution
            sensor_bp = self.world.get_blueprint_library().find(conf['blueprint_name'])
            sensor_bp.set_attribute('image_size_y', str(img_height))
            sensor_bp.set_attribute('image_size_x', str(img_width))
            
            # Sensor settings
            if conf["settings"] is not None:
                for key, value in conf['settings'].items():
                    sensor_bp.set_attribute(key, str(value))

            # RGB_hfps has a lower sensor_tick (higher frequency)
            if s_name == "RGB_hfps":
                sensor_bp.set_attribute('sensor_tick', str(self.hfps_timestep))
            else:
                sensor_bp.set_attribute('sensor_tick', str(self.timestep))

            # Attach sensor to vehicle
            camera_transform = carla.Transform(sensors_location)
            sensor = self.world.spawn_actor(sensor_bp, camera_transform, attach_to = self.hero,
                                            attachment_type = carla.libcarla.AttachmentType.Rigid)

            # Save sensor and blueprint            
            self.sensors[s_name] = sensor
            self.sensors_bp[s_name] = sensor_bp

            # Data saving directories    
            self.data_save_dirs[s_name] = self.save_dir / s_name

            if conf['visualize']:
                self.visual_dirs[s_name] = self.save_dir / 'visualizations' / s_name
            else:
                self.visual_dirs[s_name] = None

            self.sensor_queues[s_name] = queue.Queue()

            print(f'Created {s_name} of type : {sensor.type_id}')


        for dir in self.data_save_dirs.values():
            if dir is not None:
                dir.mkdir(parents=True, exist_ok=True)

        for dir in self.visual_dirs.values():
            if dir is not None:
                dir.mkdir(parents=True, exist_ok=True)


    def spawn_vehicles(self):
        batch = []
        hero = False # Hero is spawned separately
        for n, transform in enumerate(self.spawn_points):
            if n >= self.args["number_of_vehicles"]:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
                        
        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args["car_lights_on"]:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)
            self.traffic_manager.update_vehicle_lights(self.hero, True)


        # Disabling auto change lane
        vehicles = self.world.get_actors(self.vehicles_list)
        for vehicle in vehicles:
            self.traffic_manager.auto_lane_change(vehicle, False)


    def spawn_walkers(self):
        # some settings
        percentagePedestriansRunning = self.args["walkers_run_percent"]         # how many pedestrians will run
        percentagePedestriansCrossing = self.args["walkers_cross_percent"]      # how many pedestrians will walk through the road
        if self.args["seed_walkers"]:
            self.world.set_pedestrians_seed(self.args["seed_walkers"])
            random.seed(self.args["seed_walkers"])
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args["number_of_walkers"]):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])


        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # if False:
        #     self.world.wait_for_tick()
        # else:
        #     self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    def start_recording(self):
        for s_name, sensor in self.sensors.items():
            # Start all sensors
            sensor.listen(self.get_callback(s_name))

            # Temporary:
            if s_name == "DVS":
                self.dvs_prev_ts = time.time()

    def stop_recording(self):
        # Stop all sensors
        for sensor in self.sensors.values():
            sensor.stop()

    def get_relative_frame(self, frame):
        if self.first_frame is None:
            return 0
        
        return frame - self.first_frame

        
    def get_callback(self, s_name):
        callback_name = self.sensors_config[s_name]['callback']

        if not hasattr(callbacks, callback_name):
            print(f"{callback_name} not implemented, it must be defined in callbacks.py")
            print(f'The simulation will run anyway, but the "{s_name}" sensor will not work')
            callback_name = "dummy_callback"
        
        callback = getattr(callbacks, callback_name)
        weak_self = weakref.ref(self)
        return partial(callback, recorder=weak_self, sensor = s_name)
    
    def clean_buffer(self):
        with self.bgr_buffer.mutex:
            self.bgr_buffer.queue.clear()

    def record(self):
        try:
            print(f"Recording {self.args['name']} [Duration : {self.simulated_seconds} seconds] ... press ctl+c to force exit")
            self.start_recording()
            first = True
            sim_start_time = time.time()
            new_time_flag = True
            for _ in tqdm(range(self.total_frames)):
                if self.global_config["simulation"]["synchronous"]:
                    self.world.tick()
                    snapshot = self.world.get_snapshot()
                    self.current_frame = snapshot.frame
                    timestamp = snapshot.timestamp

                    if first:
                        # Save the first frame so the file numbering can start from 0
                        self.first_frame = self.current_frame
                        first = False

                    try:
                        if self.v2e_enabled:
                            if new_time_flag:
                                start_time = time.time()
                                new_time_flag = False

                            # Read a high fps frame
                            hfps_bgr = self.bgr_buffer.get(True, 1.0)

                            # Generate new events
                            hfps_luma = cv2.cvtColor(hfps_bgr, cv2.COLOR_BGR2GRAY)
                            new_events = self.v2e.generate_events(hfps_luma, timestamp.elapsed_seconds)
                            
                            end_time = time.time()

                            if new_events is not None and new_events.shape[0] > 0:
                                self.current_events = np.append(self.current_events, new_events, axis=0)
                                self.current_events = np.array(self.current_events)

                            self.bgr_count += 1
                            
                            if self.bgr_count >= self.v2e_multiplier:
                                # Save events
                                idx = self.get_relative_frame(self.current_frame)

                                end_time = time.time()
                                with open(self.save_dir/"Events_v2e_times.txt", 'a') as f:
                                    f.write("{},{}\n".format(idx, end_time - start_time))
                                new_time_flag = True

                                event_file = self.v2e_savedir / "{:06d}.npy".format(idx)
                                np.save(event_file, self.current_events)
                                
                                if self.global_config["simulation"]["v2e_visualize"]:
                                    # Save an events visualization
                                    event_vis_file = self.event_visdir / "{:06d}.png".format(idx)
                                    height, width, _ = hfps_bgr.shape
                                    event_img = events_to_event_image(self.current_events, height, width)
                                    event_img = event_img.numpy().transpose(1, 2, 0)
                                    imageio.imwrite(event_vis_file, event_img)
                                
                                # Save the last RGB frame
                                # if self.rgb_savedir is not None:
                                #     self.rgb_savedir.mkdir(parents=True, exist_ok=True)
                                #     rgb_file = self.rgb_savedir / "{:06d}.png".format(idx)
                                #     cv2.imwrite(rgb_file, hfps_bgr)

                                
                                # Reset variables
                                self.bgr_count = 0                        
                                self.current_events = np.zeros((0, 4), dtype=np.float32)
                                self.clean_buffer()
                            
                        else:
                            self.bgr_buffer.get(True, 1.0)

                    except Empty:
                        print("Some of the sensor information is missed")

                else:
                    raise NotImplementedError
                    # if first:
                    #     self.first_frame = 0
                    #     first = False
                    # self.world.wait_for_tick()

            sim_end_time = time.time()
            print("The simulation took {} seconds".format(sim_end_time - sim_start_time))
            self.stop_recording()

            # print("Average time per event : {}".format(np.mean(time_per_event_list)))

        except RuntimeError as err:
            if "time-out" in str(err):
                print(f"Time-out error encountred while recording {self.args['name']}")

        except KeyboardInterrupt:
            self.stop_recording()
            print("Quitting...")
            pass

        finally:
            # Disable Synchronous mode and reactivate rendering
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            # Destroy vehicles
            print('\ndestroying %d sensors' % len(self.sensors))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors.values()])

            # Destroy vehicles
            print('\ndestroying %d vehicles' % len(self.vehicles_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()

            # Destroy all walkers
            print('\ndestroying %d walkers' % len(self.walkers_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

            time.sleep(3)