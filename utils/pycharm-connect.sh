#curl https://download.jetbrains.com/python/pycharm-professional-2019.1.3.dmg
#ssh-keygen -b 4096 -t rsa

#on MacOS
#ssh-add -K ~/.ssh/id_rsa



cdswctl login -n `id -un` --url http://cdsw-16-launch-4.vpc.cloudera.com
cdswctl ssh-endpoint -p `id -un`/airline-sentiment -c 2 -m 4
ssh -p 3585 cdsw@localhost
