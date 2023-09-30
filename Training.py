import sys
sys.path.append(r"c:\users\henry.henry-pc\appdata\local\programs\python\python38\lib\site-packages")
import gym
from Environment import FlexSimEnv
from stable_baselines3.common.env_checker import check_env              #Import des PPO-Algorithmus
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def main():
    print("Initializing FlexSim environment...")

    # Erstellung eines FlexSim OpenAI Gym Environments
    env = FlexSimEnv(
        flexsimPath = "C:/Program Files/FlexSim 2023 Beta/program/flexsim.exe",
        modelPath = "C:/Users/Henry.HENRY-PC/Desktop/MA/Simulationsmodell/ModellX.fsm",             #WICHTIG: Modellpfad anpassen
        verbose = False,
        visible = False
        )

   
    model = PPO("MlpPolicy", env, verbose=1)                                                        #Hier geschieht das Training des PPO-Algorithmus
    print("Training model...")
    model.learn(total_timesteps=10000)
    
                                                                                                    #Speichern des Modells
    print("Saving model...")
    model.save("ModellX")

    input("Waiting for input to do some test runs...")

    
    for i in range(5):                                                                              #Auf Training aufbauende Testläufe
        env.seed(i)
        observation = env.reset()
        env.render()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")
    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()