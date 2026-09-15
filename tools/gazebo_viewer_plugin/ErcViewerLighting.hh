#ifndef ERC_VIEWER_LIGHTING_HH_
#define ERC_VIEWER_LIGHTING_HH_

#include <chrono>
#include <gz/gui/Plugin.hh>

/// Lighting correction for the local Ogre1 viewer only.
/// This is a GUI plugin, with no simulator or transport API calls.
class ErcViewerLighting final : public gz::gui::Plugin
{
  Q_OBJECT

 protected:
  void LoadConfig(const tinyxml2::XMLElement *) override;
  bool eventFilter(QObject *, QEvent *) override;

 private:
  bool reported{false};
  double maxFps{15.0};
  std::chrono::steady_clock::time_point nextFrame{};
};

#endif
