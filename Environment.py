import sys
sys.path.append(r"c:\users\henry.henry-pc\appdata\local\programs\python\python38\lib\site-packages")
import gym # Hier wird das Open-Ai-Gym Environment importiert, welches zum Training des Algorithmus gebraucht wird.
import os
import subprocess
import socket
import json
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class FlexSimEnv(gym.Env):                                          #Hier wird ein Python-Environment für FlexSim erstellt, welches als Basis für die Implementierung fungiert.
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, flexsimPath, modelPath, address='localhost', port=5005, verbose=False, visible=False):
        self.flexsimPath = flexsimPath
        self.modelPath = modelPath
        self.address = address
        self.port = port
        self.verbose = verbose
        self.visible = visible

        self.lastObservation = ""

        self._launch_flexsim()                                      #Hier wird das gewünschte FlexSim-Modell geöffnet
        
        self.action_space = self._get_action_space()                #Hier wird der Aktionsraum initialisiert
        self.observation_space = self._get_observation_space()      #Hier wird der Zustandsraum initialisiert

    def reset(self):                                                #Definition einer Reset-Funktion
        self._reset_flexsim()
        state, reward, done = self._get_observation()
        return state

    def step(self, action):                                         #Definition der iterativen Vorgehensweise: Eine Aktion resultiert in einem neuen Zustand und einer Veränderung der Belohnung
        self._take_action(action)
        state, reward, done = self._get_observation()
        info = {}
        return state, reward, done, info

    def render(self, mode='human'):                                 #Hier findet ein Update des Modells statt
        if mode == 'rgb_array':
            return np.array([0,0,0])
        elif mode == 'human':
            print(self.lastObservation)
        elif mode == 'ansi':
            return self.lastObservation
        else:
            super(FlexSimEnv, self).render(mode=mode)

    def close(self):                                                #Hier wird das FlexSim-Modell zunächst wieder geschlossen
        self._close_flexsim()
        
    def seed(self, seed=None):
        self.seedNum = seed
        return self.seedNum

    
    def _launch_flexsim(self):                                      #Hier wird FlexSim zum Training erneut geöffnet und das Training initialisiert
        if self.verbose:
            print("Launching " + self.flexsimPath + " " + self.modelPath)

        args = [self.flexsimPath, self.modelPath, "-training", self.address + ':' + str(self.port)]
        if self.visible == False:
            args.append("-maintenance")
            args.append("nogui")
        self.flexsimProcess = subprocess.Popen(args)

        self._socket_init(self.address, self.port)
    
    def _close_flexsim(self):
        self.flexsimProcess.kill()

    def _release_flexsim(self):
        if self.verbose:
            print("Sending StopWaiting message")
        self._socket_send(b"StopWaiting?")

    def _get_action_space(self):
        self._socket_send(b"ActionSpace?")
        if self.verbose:
            print("Waiting for ActionSpace message")
        actionSpaceBytes = self._socket_recv()
        
        return self._convert_to_gym_space(actionSpaceBytes)

    def _get_observation_space(self):
        self._socket_send(b"ObservationSpace?")
        if self.verbose:
            print("Waiting for ObservationSpace message")
        observationSpaceBytes = self._socket_recv()
        
        return self._convert_to_gym_space(observationSpaceBytes)

    def _reset_flexsim(self):
        if self.verbose:
            print("Sending Reset message")
        resetString = "Reset?"
        if hasattr(self, "seedNum"):
            resetString = "Reset:" + str(self.seedNum) + "?"
        self._socket_send(resetString.encode())

    def _get_observation(self):
        if self.verbose:
            print("Waiting for Observation message")
        observationBytes = self._socket_recv()
        self.lastObservation = observationBytes.decode('utf-8')
        state, reward, done = self._convert_to_observation(observationBytes)

        return state, reward, done
    
    def _take_action(self, action):
        actionStr = json.dumps(action, cls=NumpyEncoder)
        if self.verbose:
            print("Sending Action message: " + actionStr)
        actionMessage = "TakeAction:" + actionStr + "?"
        self._socket_send(actionMessage.encode())


    def _socket_init(self, host, port):
        if self.verbose:
            print("Waiting for FlexSim to connect to socket on " + self.address + ":" + str(self.port))

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((host, port))
        self.serversocket.listen();

        (self.clientsocket, self.socketaddress) = self.serversocket.accept()
        if self.verbose:
            print("Socket connected")
        
        if self.verbose:
            print("Waiting for READY message")
        message = self._socket_recv()
        if self.verbose:
            print(message.decode('utf-8'))
        if message != b"READY":
            raise RuntimeError("Did not receive READY! message")

    def _socket_send(self, msg):
        totalsent = 0
        while totalsent < len(msg):
            sent = self.clientsocket.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent = totalsent + sent

    def _socket_recv(self):
        chunks = []
        while 1:
            chunk = self.clientsocket.recv(2048)
            if chunk == b'':
                raise RuntimeError("Socket connection broken")
            if chunk[-1] == ord('!'):
                chunks.append(chunk[:-1])
                break;
            else:
                chunks.append(chunk)
        return b''.join(chunks)


    def _convert_to_gym_space(self, spaceBytes):
        paramsStartIndex = spaceBytes.index(ord('('))
        paramsEndIndex = spaceBytes.index(ord(')'), paramsStartIndex)
        
        type = spaceBytes[:paramsStartIndex]
        params = json.loads(spaceBytes[paramsStartIndex+1:paramsEndIndex])
        
        if type == b'Discrete':
            return gym.spaces.Discrete(params)
        elif type == b'Box':
            return gym.spaces.Box(np.array(params[0]), np.array(params[1]))
        elif type == b'MultiDiscrete':
            return gym.spaces.MultiDiscrete(params)
        elif type == b'MultiBinary':
            return gym.spaces.MultiBinary(params)

        raise RuntimeError("Could not parse gym space string")

    def _convert_to_observation(self, spaceBytes):
        observation = json.loads(spaceBytes)
        state = observation["state"]
        if isinstance(state, list):
            state = np.array(observation["state"])
        reward = observation["reward"]
        done = (observation["done"] == 1)
        return state, reward, done


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():

    env = FlexSimEnv(
        flexsimPath = "C:/Program Files/FlexSim 2023 Beta/program/flexsim.exe",
        modelPath = "C:/Users/Henry.HENRY-PC/Desktop/MA/Simulationsmodell/ModellX.fsm",             #WICHTIG: Hier wird der Pfad des betrachteten Modells hinterlegt
        verbose = True,
        visible = True
        )

    for i in range(2):                                                                              #Hier geschieht die iterative Anpassung der Belohnungsfunktion. Der Parameter i gibt dabei die Anzahl der Iterationen an.
        env.seed(i)
        observation = env.reset()
        env.render()
        done = False
        rewards = []
        while not done:
            action = env.action_space.sample()
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