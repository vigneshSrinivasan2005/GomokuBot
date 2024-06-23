from Gomoku import Gomoku
from pandas import DataFrame

def getMinMax(game,df):
    children=[]
    if game.game in df["State"].values:
        return (df.loc[df["State"]==game.game]).values[0][1]
    #if game is over return the value Of this position
    if(game.getWinner()!=-1):
        value=terminalStateValue(game)
        df.loc[len(df)]=[game.getState(),value]
        return value
    for move in game.getLegalMoves():
        child=game.copy()
        child.playMove(move*game.getCurPlayer())
        children.append(getMinMax(child,df))
    min=getValue(game.getCurPlayer(),children)*.9
    df.loc[len(df)]=[game.getState(),min]
    return min

#If player 1 wins value is 1, if player 2 wins value is -1, if draw value is 0
def terminalStateValue(game):
    if(game.getWinner()==1):
        return 10
    elif(game.getWinner()==2):
        return -10
    else:
        return 0

def getValue(currPlayer , children):
    if(currPlayer==1):
        return getMax(children)
    else:
        return getMin(children)

def getMax(children):
    max=-1000
    for child in children:
        if(child>max):
            max=child
    return max

def getMin(children):
    min=1000
    for child in children:
        if(child<min):
            min=child
    return min

df=DataFrame(columns=["State","Value"])
game=Gomoku(5,4)
getMinMax(game,df)
#save the df to a csv file
df.to_csv("modifiedMinMax.csv",index=False)