# shellcheck disable=SC2006
# shellcheck disable=SC2230
py_path=`which python`
flag=$2
run() {
    number=$1
    shift
    for i in $(seq "$number"); do
      # shellcheck disable=SC2068
      if [ "$flag" = 'CRS' ]; then
        $@
        $py_path distance_cal_crs_domain.py
      fi
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run "$1"