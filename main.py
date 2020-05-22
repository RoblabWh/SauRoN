import Environment, Agent, sys
from PyQt5.QtWidgets import QApplication

# Workaround for not getting error message
#def except_hook(cls, exception, traceback):
#    sys.__excepthook__(cls, exception, traceback)


def main():
    app = QApplication(sys.argv)
    env = Environment.Environment(app)
    agent = Agent.Agent()
    while not env.is_done():
        obs = env.get_observation()
        possible_actions = env.get_actions()
        action, targetLinVel = agent.predict(obs, possible_actions)
        print("Gewählte Aktion: " + str(action))
        print("Target Linear Velocity: " + str(targetLinVel))
        reward = env.step(action, targetLinVel)
        agent.total_reward += reward
    print("Total reward got: %.4f" % agent.total_reward)
    # sys.exit(app.exec_())
    # sys.excepthook = except_hook


if __name__ == '__main__':
    main()