# Ogre1 viewer lighting

This optional Gazebo GUI plugin makes the official world's white surfaces
visible in the cheaper Ogre1 viewer. It adjusts only this GUI process's local
rendering scene ambient and the four named ERC directional lights. It also caps
the spectator renderer at 15 FPS to leave resources for the simulation. It makes no
transport, simulator, world-file, material, or sensor changes. Ogre2 is ignored.

Build with Gazebo Harmonic development libraries:

```bash
cmake -S tools/gazebo_viewer_plugin -B tools/gazebo_viewer_plugin/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/gazebo_viewer_plugin/build -j2
```

Add the following plugin to the GUI config, alongside `MinimalScene` and
`GzSceneManager`:

```xml
<plugin filename="ErcViewerLighting" name="Viewer lighting">
  <max_fps>15</max_fps>
  <gz-gui>
    <property key="state" type="string">floating</property>
    <property key="width" type="double">1</property>
    <property key="height" type="double">1</property>
    <property key="showTitleBar" type="bool">false</property>
    <property key="resizable" type="bool">false</property>
  </gz-gui>
</plugin>
```

Launch only the viewer, with the existing simulation running:

```bash
GZ_GUI_PLUGIN_PATH="$PWD/tools/gazebo_viewer_plugin/build${GZ_GUI_PLUGIN_PATH:+:$GZ_GUI_PLUGIN_PATH}" \
  gz sim -g --render-engine-gui ogre --gui-config tools/gazebo_viewer.config
```

The viewer logs `ErcViewerLighting: adjusted four local Ogre1 viewer lights`
once all four world lights have appeared. Closing this viewer discards all its
local lighting overrides.

`max_fps` accepts values from 15 to 60, defaults to 15, and uses the wall clock.
Only the Ogre1 GUI rendering thread sleeps; simulator clock, physics, and sensor
rates are unchanged. Invalid values log a warning and retain the default.

When updating a plugin that is already loaded, build to a different directory
such as `build_throttled` and point `GZ_GUI_PLUGIN_PATH` there for the next
viewer. Do not overwrite a shared library while a running process maps it.
