#include "ErcViewerLighting.hh"

#include <array>
#include <cmath>
#include <thread>
#include <tinyxml2.h>
#include <gz/common/Console.hh>
#include <gz/gui/Application.hh>
#include <gz/gui/GuiEvents.hh>
#include <gz/gui/MainWindow.hh>
#include <gz/plugin/Register.hh>
#include <gz/rendering/Light.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/Scene.hh>

void ErcViewerLighting::LoadConfig(const tinyxml2::XMLElement *config)
{
  this->title = "Viewer lighting";
  if (config)
  {
    if (const auto *fps = config->FirstChildElement("max_fps"))
    {
      double value = 0.0;
      if (fps->QueryDoubleText(&value) == tinyxml2::XML_SUCCESS &&
          std::isfinite(value) && value >= 15.0 && value <= 60.0)
        this->maxFps = value;
      else
        gzwarn << "ErcViewerLighting: max_fps must be from 15 to 60; using 15"
               << std::endl;
    }
  }
  if (auto *app = gz::gui::App())
  {
    if (auto *window = app->findChild<gz::gui::MainWindow *>())
      window->installEventFilter(this);
  }
}

bool ErcViewerLighting::eventFilter(QObject *object, QEvent *event)
{
  // Render events are delivered on the viewer's rendering thread. Apply the
  // override just before drawing, after the previous world-state sync. Changes
  // affect only this process's rendering objects, never simulator entities.
  if (event->type() == gz::gui::events::PreRender::kType)
  {
    auto scene = gz::rendering::sceneFromFirstRenderEngine();
    if (scene && scene->Engine()->Name() == "ogre")
    {
      // Sleep only this spectator's rendering thread. The simulator runs in
      // a separate process, so its physics clock and camera rates are intact.
      // Reset from the actual frame start to avoid catch-up bursts after lag.
      std::this_thread::sleep_until(this->nextFrame);
      this->nextFrame = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(1.0 / this->maxFps));
      constexpr std::array<const char *, 4> names{
        "light_front", "light_back", "light_left", "light_right"};
      unsigned int found = 0;
      for (const auto *name : names)
      {
        if (auto light = scene->LightByName(name))
        {
          light->SetDiffuseColor(0.22, 0.22, 0.22);
          light->SetSpecularColor(0.02, 0.02, 0.02);
          ++found;
        }
      }
      // Wait for the ERC world's named lights before changing scene ambient.
      if (found == names.size())
      {
        scene->SetAmbientLight(0.18, 0.18, 0.18);
        if (!this->reported)
        {
          gzmsg << "ErcViewerLighting: adjusted four local Ogre1 viewer lights; "
                << "viewer limited to " << this->maxFps << " FPS"
                << std::endl;
          this->reported = true;
        }
      }
    }
  }
  return QObject::eventFilter(object, event);
}

GZ_ADD_PLUGIN(ErcViewerLighting, gz::gui::Plugin)
