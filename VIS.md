ssh mahmoud_gamal_15497@[IP]

## Tunnel
ssh -L 59000:localhost:5900 -C -N -l mahmoud_gamal_15497 [IP]

## Config
ps -C Xorg
nano /etc/X11/Xorg.conf

## Init Display
sudo xinit &

# Start VNC
export DISPLAY=:0
export XAUTHORITY=/var/run/lightdm/root/:0
sudo x11vnc -geometry 1024x768

# START GUI
startxfce4 &
