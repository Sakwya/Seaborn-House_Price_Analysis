from flask import Flask, Blueprint, request, render_template, flash, session, redirect, url_for, g, current_app
from src.analyse import hist_price_district, map_price_range
import json

bp = Blueprint('api', __name__, url_prefix='/api')


@bp.route('/', methods=['POST'])
def api():
    function = request.form["function"]
    kwargs = request.form["kwargs"]
    if function == "hist_price_district":
        return hist_price_district(kwargs)
    if function == "map_price_range":
        kwargs = json.loads(kwargs)
        return map_price_range(min_price=kwargs["min_price"], max_price=kwargs["max_price"],size=kwargs["figure_size"])
    return "None"
