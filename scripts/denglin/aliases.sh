#!/bin/bash

# pip
alias pui="pip uninstall -y"
alias pi="pip install"

# tree
alias ls2="tree -L 2 -C"
alias ls3="tree -L 3 -C"
alias ls2nb="tree -L 2 -C -I build"
alias ls3nb="tree -L 3 -C -I build"

# git
alias gc="git clone"
alias ga="git add"
alias gaa="git add . && git status"
alias gsk="git status"
alias gcm="git commit -m"
alias gcmm="git commit --amend"
alias gcmu='git commit -m "update" && git stash && git pull && git push && git stash pop'
alias gd="git diff"
alias gck="git checkout"
alias gba="git branch -a"
alias gb="git branch"
alias git_push="git stash && git pull --rebase && git push && git stash pop"
alias git_pull="git stash && git pull --rebase && git stash pop"
alias gba="git branch -a"
alias grv="git remote -v"
alias gl="git log"
alias gsb="git submodule sync && git submodule update --init --recursive"
git config --global --add safe.directory "*"
git config --global user.email $git_email
git config --global user.name $git_name

# common
alias rr="rm -rf"
alias t="touch"
alias tf="tail -f"
alias ge="grep -rni --exclude='*.log'"
alias gew="grep -rniw --exclude='*.log'"
alias genb="grep -rni --exclude-dir='build' --exclude='*.log'"
alias gewnb="grep -rniw --exclude-dir='build' --exclude='*.log'"
alias show="readlink -m"
alias nvi="nvidia-smi"
alias hd="cat /proc/driver/denglin0/hang_detect"

# cd
alias cdw="cd /workspace"
alias cds="cd /workspace/scripts/denglin"
alias cdb="cd /workspace/build/bin"

# python
alias p="python"
alias pp="pytest -v --continue-on-collection-errors -s"
alias gdp="gdb -q --args python"
alias gdpp="gdb -q --args python -m pytest -v --continue-on-collection-errors -s"

# kill
alias k1="kill %1"
alias k2="kill %2"
alias k3="kill %3"

function repeat() {
    local command="$*"
    local count=$1
    for ((i=1; i<=$count; i++))
    do
        echo "Running iteration $i"
        ${@:2}
    done
}
