class Baseline:
    def _robot_self_collision(self, solution: Sequence[float], collision_surfaces: Optional[Dict[str, np.ndarray]]=None, *, head_positions: Optional[Sequence[float]]=None, right_positions: Optional[Sequence[float]]=None) -> Optional[Tuple[str, str]]:
        """Return the first intersecting pair of nonadjacent robot links."""
        q = np.asarray(solution, dtype=float)
        right_positions = self._resolved_right_positions(right_positions)
        resolved_head = self._resolved_head_positions(head_positions)
        cache_key = joint_state_cache_key(q, right_positions, resolved_head)
        if collision_surfaces is None and cache_key in self._self_collision_cache:
            return self._self_collision_cache[cache_key]
        surfaces = self._world_collision_surfaces(q, right_positions=right_positions, head_positions=resolved_head) if collision_surfaces is None else collision_surfaces
        links = [link for link in CARRIED_COLLISION_LINKS if link in surfaces]
        watertight_links = self._watertight_collision_links()
        bounds = {link: np.asarray([np.min(surface, axis=(0, 1)), np.max(surface, axis=(0, 1))]) for (link, surface) in surfaces.items()}
        prepared_surfaces = {}
    
        def links_intersect(first_link: str, second_link: str) -> bool:
            first_bounds = bounds[first_link]
            second_bounds = bounds[second_link]
            if np.any(first_bounds[1] < second_bounds[0]) or np.any(second_bounds[1] < first_bounds[0]):
                return False
            for link in (first_link, second_link):
                if link not in prepared_surfaces:
                    prepared_surfaces[link] = PreparedTriangleMesh(surfaces[link])
            return triangle_meshes_intersect(prepared_surfaces[first_link], prepared_surfaces[second_link], first_watertight=first_link in watertight_links, second_watertight=second_link in watertight_links)
        static_links = [link for link in links if link not in LEFT_COLLISION_LINKS]
        static_cache_key = joint_state_cache_key((q[0],), right_positions, resolved_head)
        static_result = None
        if collision_surfaces is None:
            static_result = self._static_self_collision_cache.get(static_cache_key)
        if static_result is None and (collision_surfaces is not None or static_cache_key not in self._static_self_collision_cache):
            for (first_index, first_link) in enumerate(static_links):
                for second_link in static_links[first_index + 1:]:
                    if frozenset((first_link, second_link)) in DIRECTLY_CONNECTED_COLLISION_LINKS:
                        continue
                    if links_intersect(first_link, second_link):
                        static_result = (first_link, second_link)
                        break
                if static_result is not None:
                    break
            if collision_surfaces is None:
                self._static_self_collision_cache[static_cache_key] = static_result
        if static_result is not None:
            if collision_surfaces is None:
                self._self_collision_cache[cache_key] = static_result
            return static_result
        for (first_index, first_link) in enumerate(links):
            for second_link in links[first_index + 1:]:
                if first_link not in LEFT_COLLISION_LINKS and second_link not in LEFT_COLLISION_LINKS:
                    continue
                if frozenset((first_link, second_link)) in DIRECTLY_CONNECTED_COLLISION_LINKS:
                    continue
                if links_intersect(first_link, second_link):
                    result = (first_link, second_link)
                    if collision_surfaces is None:
                        self._self_collision_cache[cache_key] = result
                    return result
        if collision_surfaces is None:
            self._self_collision_cache[cache_key] = None
        return None
