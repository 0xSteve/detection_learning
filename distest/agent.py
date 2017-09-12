from lrp import Linear_Reward_Penalty as LRP


class Agent(object):
    '''An underwater mobile-agent.'''

    def __init__(self, precision, depth, channel_depth):
        '''Create a new mobile-agent with precision as a multiplier for
           channel_depth, and the depth as it's starting location.'''
        self.precision = precision  # The precision of the movement.
        self.depth = depth  # The current depth, or starting depth.
        self.starting_depth = depth  # The starting depth for each iteration.
        self.channel_depth = channel_depth  # The depth of the channel.
        self.action = 1  # The default action is do nothing.
        self.last_action = 1
        self.lrp = LRP(3)  # Use a 3-action LRP to determine direction.
        # action 0 is dive, action 1 is do-nothing, action 2 is surface.
        self.current_to = -1  # Initially the worst possible timeout.
        self.best_to = -1  # Best timeout is the shortest timeout.
        # best_to and current_to are probabilities on [0,1].
        # In order to do better movements we need to store information
        # about consecutive movement.
        self.consecutive = 0
        self.threshold = pow(2, 0)

    def move(self):
        '''Move the mobile-agent based on the action chosen by the LRP LA.
           This movement depends on the precision and channel depth.'''
        if(self.action == self.last_action):
            self.consecutive += 1
        else:
            self.consecutive = 0

        if self.action == 0:
            # DIVE
            self.dive()
        elif self.action == 2:
            # SURFACE
            self.surface()
        else:
            # In this case we have chosen to do nothing. This conserves
            # energy in the mobile-agent by not forcing it to move at all.
            pass

    def move_function(self):
        '''There are lots of optimization possibilities here. For
           now we will consider only a unimodal probability dist
           and linear movement function.'''
        if(not(self.consecutive <= self.threshold)):
            self.consecutive = 0
            print(self.consecutive)
        return pow(2, self.consecutive)

    def dive(self):
        '''This method uses the precision and channel depth to move an
           increment towards the seabed.'''
        interval = self.move_function()
        # Only move if it is possible.
        if(self.depth + interval <= self.channel_depth):
            self.depth += interval
        else:
            self.depth = self.channel_depth
        # else:
        #     self.lrp.do_penalty(self.action)

    def surface(self):
        '''This method uses the precision and channel depth to move an
           increment towards the sea surface.'''
        interval = self.move_function()
        # Only move if it is possible.
        if(self.depth - interval >= 0):
            self.depth -= interval
        else:
            self.depth = 0
        # else:
        #     self.lrp.do_penalty(self.action)

    def send(self):
        '''This function will send the depth to the environment, which will
           compute if a timeout has occurred and use the receive method to
           update the action of the LA.'''
        return (self.depth * self.precision)

    def receive(self, response):
        '''This method will simulate a test message through the environment.
           The mobile-agent, upon receiving a message, will determine it's
           next action. Here we make some assumptions about timeouts in the
           environment. This will be simulated through the use of a random
           number and the probability for the current depth of the
           mobile-agent.'''
        self.current_to = response
        # There is an important esoteric feature in this if statement.
        # Notice that the LA will reward up to two actions for the same result.
        # If the curreent best is equal to the absolute best both the
        # do-nothing and last rewarded action could be rewarded here. This is
        # intended by design, since the change in depth must be over estimated
        # before it can truly be found.
        worse = self.current_to < self.best_to
        # same_action = self.last_action == self.action
        if(worse):
            # This is the condition where the mobile-agent must tell the LRP
            # LA that a penalty is to be delivered.
            self.lrp.do_penalty(self.action)
        else:
            # Tell the LA that a reward is to be delivered.
            # This is the best value ever observed.
            self.best_to = self.current_to
            self.lrp.do_reward(self.action)

    def next_action(self):
        '''encapsulates the LRP next_action function.'''
        self.action = self.lrp.next_action()
