for ((argpos=1; argpos<$#; argpos++)); do
  if [ "${!argpos}" == "--config" ]; then
    argpos_plus1=$((argpos+1))
    config=${!argpos_plus1}
    [ ! -r $config ] && echo "$0: missing config '$config'" && exit 1
    . $config
  fi
done

while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in

    --help|-h) if [ -z "$help_message" ]; then echo "No help found." 1>&2;
      else printf "$help_message\n" 1>&2 ; fi;
      exit 0 ;;
    --*=*) echo "$0: options to scripts must be of the form --name value, got '$1'"
      exit 1 ;;

    --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`;

      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;

      oldval="`eval echo \\$$name`";
      # Work out whether we seem to be expecting a Boolean argument.
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi


      eval $name=\"$2\";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;
  *) break;
  esac
done


# Check for an empty argument to the --cmd option, which can easily occur as a
# result of scripting errors.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" 1>&2 && exit 1;


true; # so this script returns exit code 0.
