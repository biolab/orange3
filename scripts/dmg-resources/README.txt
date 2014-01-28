OSX dmg installer resources
===========================

This directory contains the binary resources to create an
Orange dmg installer.


DS_Store
	OSX Desktop Services folder settings.
	Defined the folder background selection, icon positions and size,
	Finder window position and size, ...
	This gets copied to DMGROOT/.DS_Store

VolumeIcon.icns
	Volume icon.
	This gets copied to DMGROOT/.VolumeIcon.icns

background.png
	The background folder image.
	This gets copied to DMGROOT/.background/background.png
	as specified by DS_Store.
	Note if you change the path of the background image on the
	dmg you must also update the create-dmg-installer.sh script
	so it is moved to the proper place.


Modifying the installer folder
------------------------------

The installer is nothing more then a Finder folder view
with hidden sidebar, toolbar and statusbar.

Mount an disk image in RW mode (pass the --keep-temp option to
create-dmg-installer.sh to keep the temporary uncompressed image)
and open the Show View Options.
Select the Picture background you want (from the dmg itself). Hint: use
Command-Shift-. to show hidden files in the Open dialog. From the
View Options also select the proper icon size to match the background
position indicators. Then move the .app and /Applications symlink
icons to their specified place on the background.

Resize the finder window (without the toolbar, sidebar, ...) so it
fits the background. Position the window somewhere on the top left
in the 1024, 768 rect so it can fit on small monitors.

Eject the image so .DS_Store is flushed. Then re-mount it and copy the
.DS_Store, and background image to this directory.
