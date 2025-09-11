import numpy as np
from collections import deque
from thortils.utils import euclidean_dist


class PathFollowingRewardSystem:
    """
    Reward system for training models to follow A* paths in AI2-Thor.
    Promotes staying on the optimal path and penalizes deviations.
    """

    def __init__(self, astar_path, destination, config=None):
        """
        Initialize the reward system.

        Args:
            astar_path: List of (x, y, yaw) tuples representing the A* path
            destination: Target position (x, y, z) tuple
            config: Dictionary of reward parameters
        """
        self.astar_path = astar_path
        self.destination = destination
        self.path_length = len(astar_path)

        # Default reward configuration
        default_config = {
            'path_progress_weight': 2.0,  # Reward for making progress along path
            'distance_penalty_weight': 1.0,  # Penalty for being far from path
            'backtrack_penalty': 3.0,  # Strong penalty for going backwards
            'completion_bonus': 10.0,  # Bonus for reaching destination
            'step_penalty': 0.05,  # Small penalty per step to encourage efficiency
            'path_deviation_threshold': 0.5,  # Maximum allowed distance from path
            'progress_lookahead': 3,  # How many steps ahead to look for path matching
            'smoothing_factor': 0.8  # For exponential moving average of rewards
        }

        self.config = {**default_config, **(config or {})}

        # State tracking
        self.current_path_index = 0
        self.max_path_index_reached = 0
        self.visited_positions = deque(maxlen=10)  # Track recent positions for backtracking detection
        self.last_reward = 0.0
        self.smoothed_reward = 0.0

        # Pre-compute path distances for efficiency
        self._precompute_path_metrics()

    def _precompute_path_metrics(self):
        """Pre-compute distances between consecutive path points."""
        self.path_distances = []
        for i in range(len(self.astar_path) - 1):
            dist = euclidean_dist(
                (self.astar_path[i][0], 0, self.astar_path[i][1]),
                (self.astar_path[i + 1][0], 0, self.astar_path[i + 1][1])
            )
            self.path_distances.append(dist)

    def reset(self):
        """Reset the reward system for a new episode."""
        self.current_path_index = 0
        self.max_path_index_reached = 0
        self.visited_positions.clear()
        self.last_reward = 0.0
        self.smoothed_reward = 0.0

    def calculate_reward(self, current_position, action_taken=None):
        """
        Calculate reward based on current position and path following behavior.

        Args:
            current_position: Current agent position (x, y, z) or (x, z) tuple
            action_taken: Optional action that led to this position

        Returns:
            Dictionary containing total reward and component breakdown
        """
        if len(current_position) == 3:
            agent_pos_2d = (current_position[0], current_position[2])
        else:
            agent_pos_2d = current_position

        self.visited_positions.append(agent_pos_2d)

        reward_components = {}
        total_reward = 0.0

        # 1. Path Progress Reward
        progress_reward = self._calculate_progress_reward(agent_pos_2d)
        reward_components['path_progress'] = progress_reward
        total_reward += progress_reward * self.config['path_progress_weight']

        # 2. Distance from Path Penalty
        distance_penalty = self._calculate_distance_penalty(agent_pos_2d)
        reward_components['distance_penalty'] = distance_penalty
        total_reward += distance_penalty * self.config['distance_penalty_weight']

        # 3. Backtracking Penalty
        backtrack_penalty = self._calculate_backtrack_penalty()
        reward_components['backtrack_penalty'] = backtrack_penalty
        total_reward += backtrack_penalty * self.config['backtrack_penalty']

        # 4. Completion Bonus
        completion_bonus = self._calculate_completion_bonus(agent_pos_2d)
        reward_components['completion_bonus'] = completion_bonus
        total_reward += completion_bonus

        # 5. Step Penalty (to encourage efficiency)
        step_penalty = -self.config['step_penalty']
        reward_components['step_penalty'] = step_penalty
        total_reward += step_penalty

        # 6. Smooth the reward to reduce noise
        self.smoothed_reward = (self.config['smoothing_factor'] * self.smoothed_reward +
                                (1 - self.config['smoothing_factor']) * total_reward)

        self.last_reward = total_reward

        return {
            'total_reward': total_reward,
            'smoothed_reward': self.smoothed_reward,
            'components': reward_components,
            'path_progress': self.current_path_index / self.path_length if self.path_length > 0 else 0,
            'max_progress': self.max_path_index_reached / self.path_length if self.path_length > 0 else 0
        }

    def _calculate_progress_reward(self, agent_pos):
        """Reward for making progress along the A* path."""
        if not self.astar_path:
            return 0.0

        # Find the closest point on the path within a lookahead window
        best_index = self.current_path_index
        min_distance = float('inf')

        # Look ahead in the path to find the best matching point
        lookahead_end = min(len(self.astar_path),
                            self.current_path_index + self.config['progress_lookahead'])

        for i in range(self.current_path_index, lookahead_end):
            path_point = (self.astar_path[i][0], self.astar_path[i][1])
            distance = euclidean_dist((agent_pos[0], 0, agent_pos[1]),
                                      (path_point[0], 0, path_point[1]))

            if distance < min_distance:
                min_distance = distance
                best_index = i

        # Update path index if we've made progress
        progress_made = 0.0
        if best_index > self.current_path_index:
            progress_made = best_index - self.current_path_index
            self.current_path_index = best_index
            self.max_path_index_reached = max(self.max_path_index_reached, best_index)

        # Reward proportional to progress made
        return progress_made / self.path_length if self.path_length > 0 else 0.0

    def _calculate_distance_penalty(self, agent_pos):
        """Penalty for being far from the optimal path."""
        if not self.astar_path or self.current_path_index >= len(self.astar_path):
            return 0.0

        # Distance to current path point
        current_path_point = (self.astar_path[self.current_path_index][0],
                              self.astar_path[self.current_path_index][1])
        distance = euclidean_dist((agent_pos[0], 0, agent_pos[1]),
                                  (current_path_point[0], 0, current_path_point[1]))

        # Exponential penalty that increases with distance
        if distance > self.config['path_deviation_threshold']:
            penalty = -np.exp((distance - self.config['path_deviation_threshold']) / 0.5)
        else:
            penalty = 0.0

        return penalty

    def _calculate_backtrack_penalty(self):
        """Strong penalty for revisiting recently visited positions."""
        if len(self.visited_positions) < 3:
            return 0.0

        current_pos = self.visited_positions[-1]
        penalty = 0.0

        # Check if current position is similar to recent positions
        for i, past_pos in enumerate(list(self.visited_positions)[:-1]):
            distance = euclidean_dist((current_pos[0], 0, current_pos[1]),
                                      (past_pos[0], 0, past_pos[1]))

            if distance < 0.3:  # Very close to a recent position
                # Stronger penalty for more recent revisits
                time_penalty = 1.0 / (len(self.visited_positions) - i)
                penalty += -time_penalty

        return penalty

    def _calculate_completion_bonus(self, agent_pos):
        """Large bonus for reaching the destination."""
        dest_pos = (self.destination[0], self.destination[2]) if len(self.destination) == 3 else self.destination
        distance_to_dest = euclidean_dist((agent_pos[0], 0, agent_pos[1]),
                                          (dest_pos[0], 0, dest_pos[1]))

        if distance_to_dest <= 0.5:  # Close enough to destination
            return self.config['completion_bonus']

        return 0.0

    def get_path_visualization_data(self, current_position):
        """Get data for visualizing the path and agent progress."""
        if len(current_position) == 3:
            agent_pos_2d = (current_position[0], current_position[2])
        else:
            agent_pos_2d = current_position

        return {
            'astar_path': self.astar_path,
            'current_position': agent_pos_2d,
            'current_path_index': self.current_path_index,
            'max_progress': self.max_path_index_reached,
            'visited_positions': list(self.visited_positions),
            'destination': self.destination
        }


# Integration example with your existing NavigationUtils class
class RewardAwareNavigationUtils(NavigationUtils):
    """Extended NavigationUtils with reward system integration."""

    def __init__(self):
        super().__init__()
        self.reward_system = None

    def setup_reward_system(self, target_point, reward_config=None):
        """Setup reward system after planning a path."""
        if self.last_path_gen is None:
            raise ValueError("No path generated yet. Call get_path_cost_to_target_point first.")

        destination = (target_point.x, target_point.y) if hasattr(target_point, 'x') else target_point
        self.reward_system = PathFollowingRewardSystem(
            self.last_path_gen,
            destination,
            reward_config
        )

        return self.reward_system

    def get_step_reward(self, current_position, action_taken=None):
        """Get reward for the current step."""
        if self.reward_system is None:
            raise ValueError("Reward system not initialized. Call setup_reward_system first.")

        return self.reward_system.calculate_reward(current_position, action_taken)


# Example usage and configuration
def create_reward_config_variants():
    """Different reward configurations for different training scenarios."""

    # Strict path following - heavily penalizes deviations
    strict_config = {
        'path_progress_weight': 3.0,
        'distance_penalty_weight': 2.0,
        'backtrack_penalty': 5.0,
        'completion_bonus': 15.0,
        'step_penalty': 0.1,
        'path_deviation_threshold': 0.3,
        'progress_lookahead': 2
    }

    # Flexible path following - allows some exploration
    flexible_config = {
        'path_progress_weight': 2.0,
        'distance_penalty_weight': 0.5,
        'backtrack_penalty': 2.0,
        'completion_bonus': 10.0,
        'step_penalty': 0.03,
        'path_deviation_threshold': 0.8,
        'progress_lookahead': 5
    }

    # Efficiency focused - prioritizes speed
    efficient_config = {
        'path_progress_weight': 4.0,
        'distance_penalty_weight': 1.0,
        'backtrack_penalty': 4.0,
        'completion_bonus': 20.0,
        'step_penalty': 0.2,
        'path_deviation_threshold': 0.4,
        'progress_lookahead': 3
    }

    return {
        'strict': strict_config,
        'flexible': flexible_config,
        'efficient': efficient_config
    }