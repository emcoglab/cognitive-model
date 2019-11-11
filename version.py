from git import Repo

# major.minor.rev
#   major: for published output
#   minor: for functional changes (worth of write-up)
#   rev:   will change output of model
#
# output within the same version should be considered compatible
VERSION = "0.6.1"

GIT_HASH = Repo(search_parent_directories=True).head.object.hexsha
