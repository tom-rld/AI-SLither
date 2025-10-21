
class Camera:
    def __init__(self, x, y,playerDims, winDims):
        self.x = x
        self.y = y
        self.playerDims = playerDims
        self.winDims = winDims

    def update(self, target_x, target_y, map_center=(0, 0), map_radius=None):
        """
        Met à jour la position de la caméra centrée sur la cible (le serpent).
        Ne limite plus la caméra au cercle — suit le serpent librement.
        """
        self.x = target_x - (self.winDims[0] / 2)
        self.y = target_y - (self.winDims[1] / 2)

    def translate(self, world_x, world_y):
        """
        Convertit une coordonnée du monde en coordonnée écran.
        """
        return (
            int(world_x - self.x),
            int(world_y - self.y)
        )
