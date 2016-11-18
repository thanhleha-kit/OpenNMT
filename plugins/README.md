## Plugins

This directory contains user plugins. The plugins are stored in different directories which need to contain at least a `init.lua` initialize on the model of `extension/init.lua`.

Plugins are initialized just after commandline parsing and get options from commandline.

Plugins can register function for different 'hooks' that will declared in the code as following:

<pre>
Plugins.triggerHooks('HOOKNAME', {PARAMETERS})
</pre>

Plugins can register options that display in option list when activated