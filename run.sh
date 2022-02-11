#!/bin/bash

gunicorn --workers 3 -b localhost:8675 --reload server:app --reload
