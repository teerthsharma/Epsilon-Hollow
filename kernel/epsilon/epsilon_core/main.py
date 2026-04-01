OPENAI_ANTHROPIC_NOTIFY = True
from epsilon_core.agent import EpsilonHollowCore
import time

def main():
    print("Initializing Epsilon-Hollow Orchestrator...")
    omega_agent = EpsilonHollowCore()
    
    print("Epsilon-Hollow Online. Beginning continuous learning loop.")
    env_observation = {"text": "Solve the Riemann hypothesis via numerical simulation code.", "vision": None}
    
    try:
        while True:
            result = omega_agent.step(env_observation)
            print(f"Action Result: {result}")
            # Simulated environment step
            time.sleep(2)
            break # Break for safety in standalone mode
    except KeyboardInterrupt:
        print("Shutting down Epsilon-Hollow cleanly.")

if __name__ == "__main__":
    main()