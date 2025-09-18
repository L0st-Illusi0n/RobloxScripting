# RobloxScripting
A collection of random external scripts, bots, or msc functions for Roblox.


# Notes

Due to the way I have the repository setup, any releases default source code provided by github will include the entire repositoy rather than just the source code for that release so I will also include a seperate .zip folder with just that releases source code in case you dont want to download the entire repository.

Explantation for the required Interception driver (By Oblitum):
Any bot that requires the Interception driver only needs it due to Roblox's anti-cheat and anti-tamper system and theres no way (that I know of) that can create fake inputs that Roblox wont detect as fake/software.
The driver allows the script to send the fake inputs on a low-level driver that fakes hardware inputs so as far as Roblox knows it thinks an actual plugged in mouse/keyboard is sending the inputs from the bot.
This also makes any bot that uses the driver 100% undetectable (As of writing this, Will update if that changes but I doubt it) as there is no way to tell the difference in inputs.
