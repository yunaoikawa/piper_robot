# Rollout Package

Modular framework for robot policy control and data collection.

## Architecture

The package is organized into focused modules, each handling a specific concern:

```
rollout/
├── __init__.py              # Package exports
├── camera.py                # Camera feed management and display
├── recorder.py              # Data recording and HDF5/video saving
├── episode.py               # Episode state and autonomous control logic
├── keyboard.py              # Keyboard input handling
├── manipulability.py        # Manipulability calculation
└── controller.py            # Main controller orchestrating all components
```

## Modules

### CameraFeedManager (`camera.py`)
Handles iPhone camera feed capture and live display.

**Responsibilities:**
- Connect to iPhone via Record3D
- Continuously capture RGB frames
- Display live feed with status overlay
- Thread-safe frame access

### DataRecorder (`recorder.py`)
Manages recording and saving of robot episode data.

**Responsibilities:**
- Queue-based data collection
- Save episodes to HDF5 format
- Generate MP4 videos from RGB frames
- Thread-safe recording operations

### EpisodeManager (`episode.py`)
Manages episode lifecycle and autonomous mode logic.

**Responsibilities:**
- Track episode state (active/inactive)
- Start/end episodes
- Autonomous mode: auto-start, timeout, manipulability checks
- Coordinate with recorder for data collection

### KeyboardController (`keyboard.py`)
Handles keyboard input for manual episode control.

**Responsibilities:**
- Non-blocking keyboard input
- Episode start/stop commands ('s'/'e')
- Graceful shutdown ('q')
- Terminal settings management

### ManipulabilityCalculator (`manipulability.py`)
Calculates robot manipulability from Jacobian.

**Responsibilities:**
- Fetch Jacobian via RPC
- Compute manipulability ellipsoid volume: `sqrt(det(J @ J^T))`
- Error handling

### PolicyController (`controller.py`)
Main orchestrator coordinating all components.

**Responsibilities:**
- Robot RPC communication
- ZeroMQ setup for HPC server
- Observation publishing loop
- Action request/apply loop
- Component lifecycle management

## Usage

```python
from rollout import PolicyController

# Create controller
controller = PolicyController(
    hpc_host="192.168.1.50",
    enable_recording=True,
    autonomous_mode=False,
    episode_timeout=60.0,
    manipulability_threshold=0.05
)

# Run control loop
controller.control_loop(control_rate=4)

# Clean shutdown
controller.stop()
```

Or use the provided script:

```bash
# Manual mode with recording
python cloud_inference_control_collect_v2.py --record

# Autonomous mode
python cloud_inference_control_collect_v2.py --autonomous --record

# Custom settings
python cloud_inference_control_collect_v2.py \
  --autonomous \
  --record \
  --episode-timeout 120 \
  --manipulability-threshold 0.1 \
  --rate 10
```

## Design Principles

### Separation of Concerns
Each module has a single, well-defined responsibility:
- **Camera**: Only handles camera I/O and display
- **Recorder**: Only handles data persistence
- **Episode**: Only handles episode lifecycle
- **Keyboard**: Only handles user input
- **Manipulability**: Only handles kinematics calculation
- **Controller**: Coordinates but doesn't implement domain logic

### Dependency Injection
Components are loosely coupled through constructor injection:
```python
episode_manager = EpisodeManager(recorder=recorder)
keyboard = KeyboardController(stop_event, episode_manager)
```

### Thread Safety
- Proper use of locks for shared state
- Thread-safe queues for data transfer
- Event-based synchronization

### Error Handling
- Graceful degradation on component failure
- Informative error messages
- Safe fallbacks (e.g., manipulability calculation)

## Benefits of Refactoring

1. **Maintainability**: Each file is <300 lines, focused on one thing
2. **Testability**: Components can be unit tested in isolation
3. **Reusability**: Modules can be used independently
4. **Clarity**: Clear separation of concerns
5. **Extensibility**: Easy to add new features or modify existing ones

## Migration from Original

The original `cloud_inference_control_collect.py` (~1000 lines) is now:
- **camera.py**: 170 lines
- **recorder.py**: 220 lines
- **episode.py**: 140 lines
- **keyboard.py**: 100 lines
- **manipulability.py**: 50 lines
- **controller.py**: 430 lines
- **Total**: ~1100 lines (with better organization and documentation)

The original file remains unchanged for backward compatibility.
