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

| Dataset Name                            |    size    |
|-----------------------------------------|------------|
| agenttuning_alfworld                    |    0.3K    |
| agenttuning_db                          |    0.5K    |
| agenttuning_kg                          |    0.3K    |
| agenttuning_mind2web                    |    0.1K    |
| agenttuning_os                          |    0.2K    |
| agenttuning_webshop                     |    0.4K    |
| code_feedback                           |   66.4K    |
| codeactinstruct                         |    7.1K    |
| nebius_SWE-agent-trajectories           |   13.4K    |
| nnetnav-live                            |   46.8K    |
| nnetnav-wa                              |   44.4K    |
| openhands                               |    0.1K    |
| orca_agentinstruct                      | 1046.4K    |
| SWE-Gym_OpenHands-Sampled-Trajectories  |    0.5K    |
| SWE-smith_5kTrajectories                |    5.0K    |
| synatra                                 |   99.9K    |
