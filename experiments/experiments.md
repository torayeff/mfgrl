# Experiments
## Data 1
Training steps: 2 * 500000
Mean episode reward: -46432.435436456
Decisions (1 example):
    Purchase decisions:
        Buffer (9/10): Mfg4, Mfg1, Mfg2, Mfg2, Mfg2, Mfg0, Mfg2, Mfg2, Mfg2 
    Cost: -46213.6560192324

    Comments: The agent to learn to buy only what and when cfg is needed, for example the last buy action is at demand = 1593, remaining demand time = 112

    Remaining demand: -4

    Remaining time: 44


## Data 2
Training steps: 2 * 500000
Mean episode reward: -93806.72376785279
Decisions (1 example):
    Purchase Decisions:
        Buffer (10/10): Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4
    Cost: -93694.29387664795

    Comments: The agent purchased all the configurations before starting any production.

    Remaining demand: -25

    Remaining time: 3


## Data 3
Training steps: 2 * 500000
    Mean episode reward: -28573.526899472305
    Decisions (1 example)
        Purchase Decisions:
            Buffer (10/10): Mfg2, Mfg2, Mfg2, Mfg0 (at rd=966, rt=86), Mfg0 (at rd=788, rt=60), Mfg0 (at rd=788, rt=60), Mfg0 (at rd=788, rt=60), Mfg2 (at rd=788, rt=60), Mfg2 (at rd=788, rt=60), Mfg2 (at rd=788, rt=60)
        
    Cost: -27653.55777287948

    Remaining demand: 0

    Remaining time: 7

    Comments: Initially 3 Mfg2 capacity cfgs are bought, towards the end the buffer is filled with necessary cfgs again.

## Data 4
Training steps: 2 * 500000
    Mean episode reward: -63545.30447470331
    Decisions (1 example):
        Purchase decisions:
            Buffer (10/10): Mfg4, Mfg2, Mfg4, Mfg4, Mfg2, Mfg2, Mfg4, Mfg4, Mfg4, Mfg4
    
    Cost: -64128.16091156006
    
    Remaining demand: -5
    
    Remaining time: 3

    Comments: All cfgs. are bought before starting production

## Data 5
Training steps: 2 * 500000
    Mean episode reward: -55438.75107765198
    Decisions (1 example):
        Purchase decisions:
            Buffer (10/10): Mfg4, Mfg2, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4, Mfg4
    
    Cost: -55515.173557281494
    
    Remaining demand: -7
    
    Remaining time: 1

    Comments: All cfgs. are bought before starting production