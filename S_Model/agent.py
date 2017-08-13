from lrp import LRP


class agent(object):
    '''An underwater mobile-agent.'''

    def __init__(self, precision, depth, channel_depth):
        '''Create a new mobile-agent with precision as a multiplier for
           channel_depth, and the depth as it's starting location.'''
        self.precision = precision  # The precision of the movement.
        self.depth = depth  # The current depth, or starting depth.
        self.channel_depth = channel_depth  # The depth of the channel.
        self.action = 1  # The default action is do nothing.
        self.lrp = LRP(3)  # Use a 3-action LRP to determine direction.
        # action 0 is dive, action 1 is do-nothing, action 2 is surface.

    def move(self, action):
        '''Move the mobile-agent based on the action chosen by the LRP LA.
           This movement depends on the precision and channel depth.'''
        if action == 0:
            # DIVE
            self.dive()
        elif action == 2:
            # SURFACE
            self.surface()
        else:
            # In this case we have chosen to do nothing. This conserves
            # energy in the mobile-agent by not forcing it to move at all.
            pass

    def dive(self):
        '''This method uses the precision and channel depth to move an
           increment towards the seabed.'''
        pass

    def surface(self):
        '''This method uses the precision and channel depth to move an
           increment towards the sea surface.'''
        pass

    def send(self):
        '''This function will send the depth to the environment, which will
           compute if a timeout has occurred and use the receive method to
           update the action of the LA.'''
        return self.depth

    def receive(self, is_timeout):
        '''This method will simulate a test message through the environment.
           The mobile-agent, upon receiving a message, will determine it's
           next action. Here we make some assumptions about timeouts in the
           environment. This will be simulated through the use of a random
           number and the probability for the current depth of the
           mobile-agent.'''
        if(is_timeout):
            # This is the condition where the mobile-agent must tell the LRP
            # LA that a penalty is to be delivered.
            self.lrp.do_penalty(self.action)
        else:
            # Tell the LA that a reward is to be delivered.
            self.lrp.do_reward(self.action)
