# Version number is semantic, but relates to OUTPUT, not API.
#
# publish.major.minor
#   publish: Increment for published outputs.
#   major:   Increment for functional changes worthy of write-up.
#   minor:   Increment during development to indicate output of model will change (making output incompatible).
#
# Major changes should be accompanied by a Git tag (and LNB write-up); minor ones can be but it's not necessary.
VERSION = "0.6.1"

# Any code change which doesn't change the output of the model will still alter the Git commit hash, so this can be used
# for tracking changes in output and bug-finding.
try:
    from git import Repo
    GIT_HASH = Repo(search_parent_directories=True).head.object.hexsha
except ModuleNotFoundError:
    try:
        import subprocess
        GIT_HASH = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except OSError:
        GIT_HASH = "Unknown"
