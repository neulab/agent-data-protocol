# Dataset Argument Documentation

This document outlines the input argument configurations for each dataset. The arguments are:

- `--api_env`: The API environment used for non OH default APIs.

## Dataset Argument Table

| Dataset Name                            |      --api_env       |  --is_web  |
|-----------------------------------------|----------------------|------------|
| agenttuning_alfworld                    | execute_ipython_cell |     no     |
| agenttuning_db                          |         None         |     no     |
| agenttuning_kg                          | execute_ipython_cell |     no     |
| agenttuning_mind2web                    |         None         |     no     |
| agenttuning_os                          |         None         |     no     |
| agenttuning_webshop                     | execute_ipython_cell |     no     |
| code_feedback                           |         None         |     no     |
| codeactinstruct                         | execute_ipython_cell |     no     |
| go-browse-wa                            |        browser       |     yes    |
| mind2web                                |        browser       |     yes    |
| nebius_SWE-agent-trajectories           |     execute_bash     |     no     |
| nnetnav-live                            |        browser       |     yes    |
| nnetnav-wa                              |        browser       |     yes    |
| openhands                               | execute_ipython_cell |     yes    |
| orca_agentinstruct                      |         None         |     no     |
| SWE-Gym_OpenHands-Sampled-Trajectories  |         None         |     no     |
| SWE-smith_5kTrajectories                |     execute_bash     |     no     |
| synatra                                 |        browser       |     yes    |


## Notes

- The `--api_env=None` option means the dataset does not rely on any particular API-based execution environment and thus does not have any APIs that are not defined in OpenHands.

# Dataset Statistics

| Dataset Name                            |    size    | AVG # steps |
|-----------------------------------------|------------|-------------|
| agenttuning_alfworld                    |    0.3K    |     15.5    |
| agenttuning_db                          |    0.5K    |      3.0    |
| agenttuning_kg                          |    0.3K    |     15.5    |
| agenttuning_mind2web                    |    0.1K    |      1.0    |
| agenttuning_os                          |    0.2K    |      6.9    |
| agenttuning_webshop                     |    0.4K    |      5.7    |
| code_feedback                           |   66.4K    |      4.0    |
| codeactinstruct                         |    7.1K    |      4.0    |
| go-browse-wa                            |    7.9K    |      6.8    |
| mind2web                                |    1.0K    |      9.6    |
| nebius_SWE-agent-trajectories           |   13.4K    |     16.2    |
| nnetnav-live                            |    4.6K    |     15.5    |
| nnetnav-wa                              |    3.7K    |     19.2    |
| openhands                               |    0.1K    |     18.0    |
| orca_agentinstruct                      | 1046.4K    |      1.3    |
| SWE-Gym_OpenHands-Sampled-Trajectories  |    0.5K    |     19.7    |
| SWE-smith_5kTrajectories                |    5.0K    |     29.4    |
| synatra                                 |   99.9K    |      1.0    |
