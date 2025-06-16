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
| nebius_SWE-agent-trajectories           |     execute_bash     |     no     |
| nnetnav-live                            |        browser       |     yes    |
| nnetnav-wa                              |        browser       |     yes    |
| openhands                               | execute_ipython_cell |     yes    |
| orca_agentinstruct                      | execute_ipython_cell |     no     |
| SWE-Gym_OpenHands-Sampled-Trajectories  | execute_ipython_cell |     no     |
| SWE-smith_5kTrajectories                |     execute_bash     |     no     |
| synatra                                 |        browser       |     yes    |


## Notes

- The `--api_env=None` option means the dataset does not rely on any particular API-based execution environment and thus does not have any APIs that are not defined in OpenHands.
