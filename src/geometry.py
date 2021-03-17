import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import uniform
from stl import mesh
import pickle
from typing import Optional, Tuple


class AirfoilGeometrySampler:
    """
    Implementation of an airfoil parametrization proposed in
    'An improved geometric parameter airfoil parameterization method',
    L. Xiaoqiang, H. Jun, S. Lei, L. Jing,
    Aerospace Science and Technology, 78 (2018), 241-247.

    :params:
    n_points (int): number of points used in the discretization of the upper and
        lower curve of the airfoil. The total discretization of the airfoil uses thus
        2 * n_points - 2 points.
    top_left_corner Tuple[float, float]: coordinates of
        the domain's top left corner
    bottom_right_corner Tuple[float, float]: coordinates of
        the domain's bottom right corner
    """

    def __init__(
        self,
        n_points: int = 50,
        top_left_corner: Tuple[float, float] = (-5, 5),
        bottom_right_corner: Tuple[float, float] = (5, -5),
    ) -> None:

        # Make sure the airfoil will be inside the domain
        assert (
            top_left_corner[0] < 0 and bottom_right_corner[0] > 1
        ), "The airfoil spanning the interval (0, 1) needs to be within the domain!"

        # Ranges of parameters describing the airfoil geometry.
        # Values taken from the original paper.
        # Camber parameters' ranges
        self.c_1min, self.c_1max = 0.01, 0.96
        self.c_2min, self.c_2max = 0.02, 0.97
        self.c_3min, self.c_3max = -0.074, 0.247
        self.c_4min, self.c_4max = -0.102, 0.206

        # Thickness parameters' ranges
        self.X_Tmin, self.X_Tmax = 0.2002, 0.4813
        self.T_min, self.T_max = 0.0246, 0.3227
        self.rho_min, self.rho_max = 0.175, 1.4944
        self.beta_min, self.beta_max = 0.1452, 4.8724

        self.n_points = n_points
        self.discretization_points = np.linspace(0, 1, n_points)
        # MS: added to have more points at leading edge
        self.discretization_points = self.discretization_points ** 1.9

        self.top_left_corner = top_left_corner
        self.bottom_right_corner = bottom_right_corner

    def _compute_airfoil_geometry(
        self,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        X_T: float,
        T: float,
        rho_bar: float,
        beta_bar: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes approximation of the airfoil's geometry based on the given parameters
        describing its shape.

        :params:
        c1: coefficient of camber-line-abscissa parameter equation.
            Takes values in the range (0.01, 0.96).
        c2: coefficient of camber-line-abscissa parameter equation.
            Takes values in the range (0.02, 0.97).
        c3: coefficient of camber-line-abscissa parameter equation.
            Takes values in the range (-0.074, 0.247).
        c4: coefficient of camber-line-abscissa parameter equation.
            Takes values in the range (-0.102, 0.206).
        X_T: chordwise location of maximum thickness.
            Takes values in the range (-0.2002, 0.4813).
        T: x-coordinate of the point of maximum thickness.
            Takes values in the range (0.0246, 0.3227).
        rho_bar: relative quantity of the leading edge radius.
            Takes values in the range (0.175, 1.4944).
        beta_bar: relative quantity of the trailing edge boat-tail angle.
            Takes values in the range (0.1452, 4.8724).
        :return:
        """
        rho = rho_bar * (T / X_T) ** 2
        beta = beta_bar * np.arctan(T / (1 - X_T))

        # Camber line
        x_camber = (
            3 * c1 * self.discretization_points * (1 - self.discretization_points) ** 2
            + 3
            * c2
            * (1 - self.discretization_points)
            * self.discretization_points ** 2
            + self.discretization_points ** 3
        )
        y_camber = (
            3 * c3 * self.discretization_points * (1 - self.discretization_points) ** 2
            + 3
            * c4
            * (1 - self.discretization_points)
            * self.discretization_points ** 2
        )

        # Assemble linear system to solve for thickness parameters
        system_matrix = np.array(
            [
                [np.sqrt(X_T), X_T, X_T ** 2, X_T ** 3, X_T ** 4],
                [0.5 / np.sqrt(X_T), 1, 2 * X_T, 3 * X_T ** 2, 4 * X_T ** 3],
                [0.25, 0.5, 1, 1.5, 2],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        right_hand_side = np.array([T, 0, -np.tan(beta / 2), np.sqrt(2 * rho), 0])

        t = np.linalg.solve(system_matrix, right_hand_side)

        thickness = (
            t[0] * np.sqrt(x_camber)
            + t[1] * x_camber
            + t[2] * x_camber ** 2
            + t[3] * x_camber ** 3
            + t[4] * x_camber ** 4
        )

        upper_curve = y_camber + 0.5 * thickness
        lower_curve = y_camber - 0.5 * thickness

        return x_camber, upper_curve, lower_curve

    def sample_airfoil_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function samples eight parameters describing the airfoil's geometry,
        and computes the discretizations of the airfoil's profile.

        :return:
        x_camber: x-coeficients of points used in the discretization.
        upper_curve: y-coefficients of the upper profile of the sampled airfoil.
            The discretization is evaluated at x_camber points.
        lower_curve: y-coefficients of the lower profile of the sampled airfoil.
            The discretization is evaluated at x_camber points.
        """

        c1 = uniform(self.c_1min, self.c_1max)
        c2 = uniform(self.c_2min, self.c_2max)
        c3 = uniform(self.c_3min, self.c_3max)
        c4 = uniform(self.c_4min, self.c_4max)

        X_T = uniform(self.X_Tmin, self.X_Tmax)
        T = uniform(self.T_min, self.T_max)
        rho_bar = uniform(self.rho_min, self.rho_max)
        beta_bar = uniform(self.beta_min, self.beta_max)

        x_camber, upper_curve, lower_curve = self._compute_airfoil_geometry(
            c1=c1, c2=c2, c3=c3, c4=c4, X_T=X_T, T=T, rho_bar=rho_bar, beta_bar=beta_bar
        )
        return x_camber, upper_curve, lower_curve


    def _make_stl(self, x_camber, upper_curve, lower_curve):
        """
        This function creates STL data structure for geometry.

        :return:
        stl_mesh: a 3D shell mesh of profile expanding from -0.2 to 0.2 in
            z-direction
        """
        L = len(x_camber)
        
        # we have two triangles per one point, except the last one, for each side
        data = np.zeros(2*2*(L-1), dtype=mesh.Mesh.dtype)

        I = 0
        for i in range(L-1):
            px1 = x_camber[i]
            px2 = x_camber[i+1]
            py1 = upper_curve[i]
            py2 = upper_curve[i+1]
            data["vectors"][I] = np.array([[px1, py1, -0.2], [px1, py1, 0.2], [px2, py2, -0.2]])
            data["vectors"][I+1] = np.array([[px1, py1, 0.2], [px2, py2, 0.2], [px2, py2, -0.2]])
            I += 2

        for i in range(L-1):
            px1 = x_camber[i]
            px2 = x_camber[i+1]
            py1 = lower_curve[i]
            py2 = lower_curve[i+1]
            data["vectors"][I] = np.array([[px1, py1, -0.2], [px2, py2, -0.2], [px1, py1, 0.2]])
            data["vectors"][I+1] = np.array([[px1, py1, 0.2], [px2, py2, -0.2], [px2, py2, 0.2]])
            I += 2
        
        stl_mesh = mesh.Mesh(data.copy())
        stl_mesh.update_normals()
        
        return stl_mesh
        

        
    def create_airfoil_geometry(
        self, dir_to_save: str, filename: str = "geometry"
    ) -> None:
        """
        End-to-end function sampling an airfoil's profile, triangulating it,
        and saving the geometry in .stl formats.

        :params:
        dir_to_save: directory, in which the geometry will be stored.
        filename: name of the geometry files. The default value is "geometry".
            The function saves <filename>.stl file.
        """

        x_camber, upper_curve, lower_curve = self.sample_airfoil_geometry()
 
        # create STL mesh of geometry:
        stl_mesh = self._make_stl(x_camber, upper_curve, lower_curve)
 
 
        # Create directory for saving the data
        try:
            os.makedirs(dir_to_save)
        except OSError:
            print(f"Directory {dir_to_save} already exists!")

        # Saving geometry as .p and .stl file
        path_to_save_stl = os.path.join(dir_to_save, f"{filename}.stl")
        path_to_save_pickle = os.path.join(dir_to_save, f"{filename}.p")

        stl_mesh.save(path_to_save_stl)

        with open(path_to_save_pickle, 'wb') as fp:
            pickle.dump([x_camber, upper_curve, lower_curve], fp)


    @staticmethod
    def plot_airfoil(
        x_camber, upper_curve, lower_curve, filename: Optional[str] = None
    ) -> None:
        """
        Function visualizing the provided airfoil's profile.

        :params:
        x_camber: x-coeficients of points used in the discretization.
        upper_curve: y-coefficients of the upper profile of the sampled airfoil.
            The discretization is evaluated at x_camber points.
        lower_curve: y-coefficients of the lower profile of the sampled airfoil.
            The discretization is evaluated at x_camber points.
        filename: name of the image file, in which the visualization should
            be saved. If None value is provided, then no image is saved.
            This parameter is optional and by default equal to None.
        """

        plt.figure()
        plt.plot(x_camber, lower_curve)
        plt.plot(x_camber, upper_curve)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(0, 1)
        plt.ylim(-0.25, 0.25)
        if filename is not None:
            plt.savefig(filename)
        plt.show()
