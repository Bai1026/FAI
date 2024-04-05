# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

"""
Reference:
https://github.com/karlapalem/UC-Berkeley-AI-Pacman-Project/blob/master/multiagent/multiAgents.py
https://chat.openai.com/c/ec864d67-a63e-4b21-9767-0cfa3dea534f
"""

""" For Q1"""
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Win condition should always be prioritized
        if successorGameState.isWin():
            return float('inf')

        foodList = newFood.asList()
        foodDistance = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0

        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        minGhostDistance = min(ghostDistances) if ghostDistances else float('inf')

        score = successorGameState.getScore()

        # Simple adjustments based on game state analysis
        # Reduce score as distance to nearest food increases (encourage eating food)
        score -= 2 * foodDistance

        # Adjust based on ghost proximity
        # If ghosts are scared, being closer is less risky
        scared = min(newScaredTimes) > 0
        if not scared and minGhostDistance < 2:
            score -= 200  # Avoid ghosts when they are not scared

        # If the action is stopping, apply a minor penalty
        if action == Directions.STOP:
            score -= 10

        # Encourage eating food by increasing score when food count decreases
        if len(foodList) < len(currentGameState.getFood().asList()):
            score += 100

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


"""For Q2"""
class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def max_level(game_state, depth):
            """Used only for Pacman agent hence agentIndex is always 0."""
            curr_depth = depth + 1
            if game_state.isWin() or game_state.isLose() or curr_depth == self.depth:  # Terminal test
                return self.evaluationFunction(game_state)
            max_value = float('-inf')
            actions = game_state.getLegalActions(0)
            for action in actions:
                successor = game_state.generateSuccessor(0, action)
                max_value = max(max_value, min_level(successor, curr_depth, 1))
            return max_value

        def min_level(game_state, depth, agent_index):
            """For all ghosts."""
            min_value = float('inf')
            if game_state.isWin() or game_state.isLose():  # Terminal test
                return self.evaluationFunction(game_state)
            actions = game_state.getLegalActions(agent_index)
            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)
                if agent_index == (game_state.getNumAgents() - 1):
                    min_value = min(min_value, max_level(successor, depth))
                else:
                    min_value = min(min_value, min_level(successor, depth, agent_index + 1))
            return min_value

        # Root level action selection
        actions = gameState.getLegalActions(0)
        current_score = float('-inf')
        return_action = None
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = min_level(next_state, 0, 1)  # Next level is a min level
            if score > current_score:  # Choose the action which maximizes the score
                return_action = action
                current_score = score
        return return_action


"""For Q3"""
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3).
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """
        
        def max_level(game_state, depth, alpha, beta):
            """Used only for Pacman agent (agentIndex is always 0)."""
            curr_depth = depth + 1
            if game_state.isWin() or game_state.isLose() or curr_depth == self.depth:  # Terminal test
                return self.evaluationFunction(game_state)
            
            max_value = float('-inf')
            actions = game_state.getLegalActions(0)
            for action in actions:
                successor = game_state.generateSuccessor(0, action)
                max_value = max(max_value, min_level(successor, curr_depth, 1, alpha, beta))
                if max_value > beta:
                    return max_value  # Pruning
                alpha = max(alpha, max_value)
            return max_value
        
        def min_level(game_state, depth, agent_index, alpha, beta):
            """For all ghosts."""
            min_value = float('inf')
            if game_state.isWin() or game_state.isLose():  # Terminal test
                return self.evaluationFunction(game_state)
            
            actions = game_state.getLegalActions(agent_index)
            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)
                if agent_index == (game_state.getNumAgents() - 1):
                    min_value = min(min_value, max_level(successor, depth, alpha, beta))
                else:
                    min_value = min(min_value, min_level(successor, depth, agent_index + 1, alpha, beta))
                
                if min_value < alpha:
                    return min_value  # Pruning
                beta = min(beta, min_value)
            return min_value

        # Initial call to Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        current_score = float('-inf')
        return_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            score = min_level(next_state, 0, 1, alpha, beta)
            if score > current_score:
                return_action = action
                current_score = score
            
            if score > beta:
                return return_action  # Early return based on pruning at root
            
            alpha = max(alpha, score)
        return return_action


"""For Q4"""
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            # totalmaxvalue = 0
            # numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,expectLevel(successor,currDepth,1))
            return maxvalue
        
        #For all ghosts.
        def expectLevel(gameState,depth, agentIndex):
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor,depth)
                else:
                    expectedvalue = expectLevel(successor,depth,agentIndex+1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return  0
            return float(totalexpectedvalue)/float(numberofactions)
        
        #Root level action.
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a expect level. Hence calling expectLevel for successors of the root.
            score = expectLevel(nextState,0,1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


"""For Q5"""
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function (question 5).

    DESCRIPTION:
    This function evaluates the game state based on the distance to food, ghost states (including scared times),
    and the number of power pellets available. It aims to encourage Pacman to eat food, avoid ghosts unless they
    are scared, and collect power pellets strategically.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # Using list comprehension for more Pythonic code
    foodList = newFood.asList()
    # Import statement moved to top (it's assumed to be at the module level for this snippet)
    foodDistances = [manhattanDistance(newPos, pos) for pos in foodList if foodList]

    ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
    ghostDistances = [manhattanDistance(newPos, pos) for pos in ghostPositions]

    # Number of power pellets available
    numberOfPowerPellets = len(currentGameState.getCapsules())

    # Initial score based on current game state
    score = currentGameState.getScore()

    # Adjust score based on food distance
    reciprocalFoodDistance = 1.0 / sum(foodDistances) if foodDistances else 0
    score += reciprocalFoodDistance

    # Additional score components
    numberOfFoods = len(foodList)
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum(ghostDistances)
    
    # Adjust score based on game state conditions
    if sumScaredTimes > 0:
        score += sumScaredTimes - numberOfPowerPellets - sumGhostDistance
    else:
        score += sumGhostDistance + numberOfPowerPellets

    # Including the count of remaining food as a negative score to encourage eating
    score -= numberOfFoods

    return score

# Abbreviation
better = betterEvaluationFunction