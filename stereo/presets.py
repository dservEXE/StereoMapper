PRESETS = {
    "Зона 1×1 м · с 15 см": dict(width="1280",height="720",fps="30",backend="AUTO",
        quality="Сбалансированная",processing_width="960",disparities="192",max_depth="0.8",
        near_priority=True,near_minimum_cm="10",wide_view=True,temporal=True,voxel_mm="1",
        map_stride="1",preview_points="250000",estimate_surfaces=False),
    "Зона 1×1 м · с 30 см": dict(width="1280",height="720",fps="30",backend="AUTO",
        quality="Сбалансированная",processing_width="1280",disparities="256",max_depth="1.2",
        near_priority=False,near_minimum_cm="15",wide_view=True,temporal=True,voxel_mm="1",
        map_stride="1",preview_points="250000",estimate_surfaces=False),
    "Комната": dict(width="1280",height="720",fps="30",backend="AUTO",
        quality="Сбалансированная",processing_width="960",disparities="128",max_depth="5",
        near_priority=False,near_minimum_cm="15",wide_view=True,temporal=True,voxel_mm="10",
        map_stride="2",preview_points="250000",estimate_surfaces=False),
    "Быстрый обзор": dict(width="1280",height="720",fps="30",backend="AUTO",
        quality="Сбалансированная",processing_width="640",disparities="96",max_depth="5",
        near_priority=False,near_minimum_cm="15",wide_view=True,temporal=True,voxel_mm="5",
        map_stride="2",preview_points="250000",estimate_surfaces=False),
}
for preset in PRESETS.values():
    preset["ghost_filter"]="Обычная"
