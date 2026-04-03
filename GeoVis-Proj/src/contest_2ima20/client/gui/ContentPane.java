/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package contest_2ima20.client.gui;

import contest_2ima20.core.problem.Viewable;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

import java.util.ArrayList;
import java.util.List;
import nl.tue.geometrycore.geometry.Vector;
import nl.tue.geometrycore.geometry.linear.Rectangle;
import nl.tue.geometrycore.geometryrendering.GeometryPanel;

/**
 *
 * @author Wout
 */
public class ContentPane extends GeometryPanel {

    private Viewable view;
    private List<ViewListener> viewListeners = new ArrayList();

    public ContentPane() {
        super();

        this.addComponentListener(new ComponentAdapter() {
            public void componentHidden(ComponentEvent e) {
                /* code run when component hidden*/
            }

            public void componentShown(ComponentEvent e) {
                zoomToFit();
            }
        });

    }

    public Viewable getViewable() {
        return view;
    }

    public void setView(Viewable view, boolean reset) {
        this.view = view;
        updateViewListeners(view);
        if (reset) {
            zoomToFit();
        } else {
            repaint();
        }
    }

    @Override
    protected void drawScene() {
        if (view != null) {
            view.draw(this);
        }
    }

    @Override
    public Rectangle getBoundingRectangle() {
        if (view == null) {
            return null;
        } else {
            return view.getBoundingBox();
        }
    }

    @Override
    protected void mousePress(Vector loc, int button, boolean ctrl, boolean shift, boolean alt) {

    }

    @Override
    protected void keyPress(int keycode, boolean ctrl, boolean shift, boolean alt) {

    }

    public void addViewListener(ViewListener v) {
        if (v != null) {
            viewListeners.add(v);
        }
    }

    public void removeViewListener(ViewListener v) {
        if (v != null) {
            viewListeners.remove(v);
        }
    }

    private void updateViewListeners(Viewable v) {
        for (ViewListener l : viewListeners) {
            l.updateView(v);
        }
    }
}
