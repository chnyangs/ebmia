# shellcheck disable=SC2006
# shellcheck disable=SC2230
py_path=`which python`
flag=$2
run() {
    number=$1
    shift
    for i in $(seq "$number"); do
      # shellcheck disable=SC2068
      if [ "$flag" = 'CLU' ]; then
        $@
        $py_path 'attack_cluster_based_crs_domain.py'
      elif [ "$flag" = 'ITE' ]; then
        # shellcheck disable=SC2068
        $@
        $py_path 'attack_iterative_based_crs_domain.py'
      elif [ "$flag" = 'DEF' ]; then
        $@
        $py_path 'attack_default_crs_domain.py'
      fi
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"