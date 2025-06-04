# JSONL Format Compliance Checklist

All `raw_to_standardized.py` files should read in jsonl and write out jsonl. Below is the status of each file:

| Dataset | Reads JSONL | Writes JSONL | Status |
|---------|-------------|--------------|--------|
| orca_agentinstruct | ✅ | ✅ | Fixed ✓ |
| omniact | ✅ | ✅ | Already Compliant ✓ |
| codeactinstruct | ✅ | ✅ | Fixed ✓ |
| webarena_successful | ✅ | ✅ | Fixed ✓ |
| SWE-Gym_OpenHands-Sampled-Trajectories | ✅ | ✅ | Already Compliant ✓ |
| androidcontrol | ✅ | ✅ | Already Compliant ✓ |
| turkingbench | ✅ | ✅ | Fixed ✓ |
| screenagent | ✅ | ✅ | Already Compliant ✓ |
| openhands | ✅ | ✅ | Fixed ✓ |
| SWE-smith_5kTrajectories | ✅ | ✅ | Already Compliant ✓ |
| android_in_the_wild | ✅ | ✅ | Already Compliant ✓ |
| agenttuning | ✅ | ✅ | Already Compliant ✓ |
| mind2web | ✅ | ✅ | Already Compliant ✓ |
| synatra | ✅ | ✅ | Fixed ✓ |
| code_feedback | ✅ | ✅ | Already Compliant ✓ |
| nebius_SWE-agent-trajectories | ✅ | ✅ | Fixed ✓ |
| llava_plus | ✅ | ✅ | Fixed ✓ |
| wonderbread | ✅ | ✅ | Fixed ✓ |
| weblinx | ✅ | ✅ | Fixed ✓ |
| nnetnav | ✅ | ✅ | Already Compliant ✓ |
| eto | ✅ | ✅ | Fixed ✓ |

## Summary
- All files correctly read from JSONL format (line by line from stdin)
- All files now correctly write to JSONL format (line by line to stdout)
- 11 files were already compliant
- 10 files were fixed to output JSONL instead of a single JSON array
