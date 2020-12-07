"""
Module providing the functionality to build Flask WSGI applications for FIS
"""

from flask import Flask, request
from flask_restx import Api, Resource, fields

from . import __version__
from . import FIS, FloatDomain, CategoricalDomain


def _variable_to_field(v):
    """Transform a FuzzyVariable into a restx field"""
    if isinstance(v.domain, FloatDomain):
        a, b = v.domain.min, v.domain.max
        f = fields.Float(description=v.name, required=True, min=a, max=b, example=(a + b) / 2)
    elif isinstance(v.domain, CategoricalDomain):
        raise NotImplementedError
    else:
        raise ValueError("Unknown domain for variable %s" % v)

    return v.name, f


class FISFlask:
    """A wrapper for a Flask application serving a FIS. Use its app attribute to access it"""

    def __init__(self, import_name, fis, serve_model_info=True):
        """

        Args:
            import_name (str): Name of the application package to pass to Flask
            fis (FIS): The fuzzy inference system
            serve_model_info (bool): Whether to provide the description of the FIS as a json.
        """
        self.app = Flask(import_name)

        self.fis = fis

        self.api = Api(app=self.app, version=__version__, title=import_name, description="A zadeh-generated FIS API")
        self.api_namespace = self.api.namespace("api", description="Api for " + import_name)

        self.app.config.update(
            ERROR_404_HELP=False,  # No "but did you mean" messages
            RESTX_MASK_SWAGGER=False,
        )

        self.models = {}

        self.models["version"] = self.api.model('Deployment configuration', {
            'version': fields.String(description="zadeh version", example=__version__),
        })

        @self.api_namespace.route('/')
        class Version(Resource):
            @self.api.marshal_with(self.models["version"], code=200, description='OK')
            def get(self):
                """Return the server version"""
                return {"version": __version__
                        }

        if serve_model_info:
            @self.api_namespace.route('/info')
            class Info(Resource):
                def get(self):
                    """Get the FIS description in zadeh-compatible format"""
                    return fis._get_description()

        self.models["input"] = self.api.model('Input', dict([_variable_to_field(v) for v in fis.variables]))

        @self.api_namespace.route('/predict/')
        class Predict(Resource):

            @self.api.expect(self.models["input"])
            def post(self):
                """Evaluate the fuzzy model for a single input"""
                return fis.get_crisp_output(request.json)


def main():
    import sys
    import os
    if len(sys.argv) != 2:
        print("Usage: %s <fis file>" % sys.argv[0])
        return

    path = sys.argv[1]

    fis_flask = FISFlask(os.path.basename(path), FIS.load(path))
    app = fis_flask.app

    app.run(debug=False, host='0.0.0.0')


if __name__ == '__main__':
    main()
