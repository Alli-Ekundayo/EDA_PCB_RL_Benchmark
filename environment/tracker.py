import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import matplotlib.animation as animation

class PlacementTracker:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.frames = deque(maxlen=max_steps)
        self.ratsnest_frames = deque(maxlen=max_steps)
        self.rewards = deque(maxlen=max_steps)
        self.hpwls = deque(maxlen=max_steps)
        
    def record_step(self, board_state, ratsnest_map, reward, hpwl):
        # We store a simplified representation for the video
        self.frames.append(board_state.copy())
        self.ratsnest_frames.append(ratsnest_map.copy())
        self.rewards.append(reward)
        self.hpwls.append(hpwl)
        
    def reset(self):
        self.frames.clear()
        self.ratsnest_frames.clear()
        self.rewards.clear()
        self.hpwls.clear()
        
    def save_video(self, output_path, fps=10):
        if not self.frames:
            print("No frames to save.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initial plots
        im1 = ax1.imshow(self.frames[0], cmap='magma', origin='lower')
        ax1.set_title("Component Placement (SDF)")
        
        im2 = ax2.imshow(self.ratsnest_frames[0], cmap='viridis', origin='lower')
        ax2.set_title("Ratsnest Density")
        
        def update(i):
            im1.set_data(self.frames[i])
            im2.set_data(self.ratsnest_frames[i])
            fig.suptitle(f"Step {i} | HPWL: {self.hpwls[i]:.2f}")
            return im1, im2

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), blit=True)
        
        # Fallback to GIF if ffmpeg is missing
        if output_path.endswith('.mp4'):
            output_path = output_path.replace('.mp4', '.gif')
            
        try:
            ani.save(output_path, writer='pillow', fps=fps)
        except Exception as e:
            print(f"Failed to save video: {e}")
            
        plt.close()
        print(f"Animation saved to {output_path}")
